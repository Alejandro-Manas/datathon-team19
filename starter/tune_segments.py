"""Evoluciona hijas de "electropapis v1" especializadas en cada tipo de cliente (en local, con el histórico).

    python starter/tune_segments.py                   # 60 minutos
    python starter/tune_segments.py --minutes=180     # 3 horas
    python starter/tune_segments.py --n=150000        # visitas por grupo (por defecto 100.000: 200.000 en total)

Datos:
  - shop.history_events (12 M eventos): se usan enteros, dentro de RawTree, para calcular las señales
    (co-compras, vista→compra, carrusel→compra, popularidad, historial de cada cliente...).
  - shop.history_sessions (2 M visitas al carrusel): 2 grupos disjuntos de visitas de la última semana del
    histórico (A para buscar, B para confirmar). Las señales se calculan solo con datos anteriores a esa semana.

Tipos de cliente (igual que en recommender_v1.segment(), con una aproximación para "nuevo"):
  nuevo       sin compras anteriores en el histórico (en vivo: is_new_user)
  sin_vistas  habitual sin vistas recientes
  producto    habitual con vistas, en ficha de producto
  categoria   habitual con vistas, en una categoría
  ("ve un producto nuevo" no existe en el histórico: en vivo usa la variante de "nuevo")

Para cada tipo, en rondas:
  - Parte de los pesos de electropapis v1 y prueba hijas (1-3 pesos cambiados: ×2, ÷2, 0...).
  - Mide cada hija con SNIPS_buy (compras estimadas corrigiendo el sesgo del carrusel antiguo) en el grupo A
    de ese tipo; si mejora a la mejor, la confirma en el grupo B. Solo se acepta si gana en los dos.
  - No toca "pop" ni "quality": en vivo vimos que el histórico sobrevalora los best sellers.
Resultado: starter/segment_weights.json. recommender_v1.py lo carga al arrancar y añade una variante
"ep-seg-<tipo>" por tipo, que solo compite con los clientes de su tipo. Registro: starter/tuning_segments_log.csv
"""
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import recommender_v1 as v1  # noqa: E402
from eval_v1 import evaluate  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "segment_weights.json"
LOG = HERE / "tuning_segments_log.csv"
MINUTES = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--minutes=")), 60))
N_EACH = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 100000))
MIN_GAIN = 0.0005
SEGS = ["nuevo", "sin_vistas", "producto", "categoria"]
FROZEN = {"pop", "quality"}                     # el histórico engaña con estas
GRID = {"buy_after_view": [0, 0.5, 1, 2, 3, 6, 10], "carousel": [0, 0.5, 1, 2, 4, 8],
        "coview": [0, 0.15, 0.3, 0.6, 1, 2], "self": [0, 0.5, 1, 2, 4, 8], "subcat": [0, 0.3, 0.75, 1.5, 3, 6],
        "copurchase": [0, 2.5, 5, 10, 20, 40]}


def load_group(cut: str, part: int) -> list[dict]:
    """Un grupo de visitas del periodo de test, descargado en trozos."""
    out: list[dict] = []
    chunks = 8
    for i in range(chunks):
        for attempt in range(3):
            try:
                out += v1.query(f"""
SELECT user_id::String AS user_id, context.page::String AS page, recent_views, shown, propensity,
       ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE ts >= '{cut}' AND cityHash64(request_id::String) % 2 = {part}
  AND cityHash64(request_id::String, 7) % {chunks} = {i}
LIMIT {N_EACH // chunks}""", 900)
                break
            except Exception as e:
                print(f"    trozo {i + 1}/{chunks} falló ({e}); reintento")
                time.sleep(5)
    return out


def seg_of(r: dict, m) -> str:
    if r["user_id"] not in m.user_bought:
        return "nuevo"
    if not [v for v in (r["recent_views"] or []) if isinstance(v, dict) and v.get("product_id")]:
        return "sin_vistas"
    return "categoria" if r["page"] == "category" else "producto"


def score(m, rows, w) -> float:
    return evaluate(lambda r: m.recommend(r["recent_views"], r["user_id"], r["page"], w), rows)["SNIPS_buy"]


def mutate(best: dict, rng: random.Random) -> dict:
    w = dict(best)
    for k in rng.sample(list(GRID), rng.randint(1, 3)):
        r = rng.random()
        if r < 0.35:
            w[k] = round(w[k] * 2 if w[k] else rng.choice(GRID[k][1:]), 3)
        elif r < 0.7:
            w[k] = round(w[k] / 2, 3)
        elif r < 0.8:
            w[k] = 0.0
        else:
            w[k] = rng.choice(GRID[k])
    return w


def main() -> None:
    rng = random.Random()
    last_day = v1.query("SELECT toString(toDate(max(ts::DateTime64(3)))) AS d FROM shop.history_sessions", 120)[0]["d"]
    cut = str(dt.date.fromisoformat(last_day) - dt.timedelta(days=7))
    print(f"Señales con datos < {cut} · visitas de evaluación >= {cut} · {MINUTES:g} minutos")

    m = v1.Model()
    m.build_history(until=cut)
    catalog = v1.query("SELECT product_id::String AS product_id, subcategory::String AS subcategory, "
                       "true AS is_active FROM shop.catalog", 60)
    m.refresh_catalog(catalog_rows=catalog, out_of_stock=set())

    print("Descargando visitas del histórico...")
    A, B = load_group(cut, 0), load_group(cut, 1)
    groups = {sg: ([r for r in A if seg_of(r, m) == sg], [r for r in B if seg_of(r, m) == sg]) for sg in SEGS}
    for sg, (a, b) in list(groups.items()):
        print(f"  {sg:<11} A: {len(a):>6} visitas ({sum(1 for r in a if r['purchased'])} con compra) · B: {len(b):>6}")
        if min(len(a), len(b)) < 300:
            print(f"  {sg:<11} → muy pocas visitas: se salta (usará las variantes generales)")
            del groups[sg]

    state = {}
    for sg, (a, b) in groups.items():
        sa, sb = score(m, a, v1.EP), score(m, b, v1.EP)
        state[sg] = {"W": dict(v1.EP), "A": sa, "B": sb, "EP_A": sa, "EP_B": sb}
        print(f"  punto de partida (electropapis v1) en {sg}: A={sa:.4f} B={sb:.4f}")

    new_log = not LOG.exists()
    log = open(LOG, "a", newline="")
    wr = csv.writer(log)
    if new_log:
        wr.writerow(["hora", "tipo", "prueba", "SNIPS_A", "SNIPS_B", "aceptada", "W"])

    def save():
        OUT.write_text(json.dumps({
            "found_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": {sg: {"W": {k: v for k, v in s["W"].items() if k not in FROZEN},
                              "SNIPS_A": s["A"], "SNIPS_B": s["B"], "electropapis_A": s["EP_A"],
                              "electropapis_B": s["EP_B"], "mejora": s["B"] > s["EP_B"] + MIN_GAIN}
                         for sg, s in state.items()}}, indent=1))

    end = time.time() + MINUTES * 60
    i = 0
    while time.time() < end:
        for sg in groups:                                 # una prueba por tipo en cada ronda
            if time.time() >= end:
                break
            i += 1
            a, b = groups[sg]
            st = state[sg]
            cand = mutate(st["W"], rng)
            sa = score(m, a, cand)
            sb, ok = "", False
            if sa > st["A"] + MIN_GAIN:
                sb = score(m, b, cand)
                if sb > st["B"] + MIN_GAIN:
                    ok = True
                    st.update(W=cand, A=sa, B=sb)
                    save()
            wr.writerow([time.strftime("%H:%M:%S"), sg, i, sa, sb, ok, json.dumps(cand)])
            log.flush()
            print(f"{time.strftime('%H:%M:%S')} {sg:<11} prueba {i:>4}  A={sa:.4f}"
                  f"{'' if sb == '' else f' B={sb:.4f}'}   mejor A={st['A']:.4f} B={st['B']:.4f} "
                  f"(madre B={st['EP_B']:.4f}){'  ★ NUEVA MEJOR' if ok else ''}")

    save()
    print("\nResultado por tipo de cliente (grupo B, el de confirmación):")
    for sg, s in state.items():
        gain = (s["B"] / s["EP_B"] - 1) if s["EP_B"] else 0
        print(f"  {sg:<11} electropapis v1={s['EP_B']:.4f}  →  ep-seg-{sg}={s['B']:.4f}  ({gain:+.0%})")
    print(f"\nGuardado en {OUT}. Para usarlo en vivo: súbelo con git y reinicia recommender_v1.py.")


if __name__ == "__main__":
    main()
