"""Afinador automático de pesos (terminal 2). Se deja corriendo y mejora solo.

    python starter/tune.py                  # 60 minutos
    python starter/tune.py --minutes=600    # toda la noche

Cómo aprende de sus propias pruebas:
  - Parte de los pesos actuales (o de best_weights.json si ya existe).
  - En cada prueba, el 75 % de las veces modifica 1-3 pesos de la MEJOR combinación encontrada
    (multiplicar/dividir por 2, poner a 0...): explota lo que ya funciona. El 25 % restante prueba
    una combinación totalmente nueva: explora, para no quedarse atascado.
  - Mide cada candidata en un grupo A de visitas del histórico (SNIPS_buy = compras estimadas).
    Si supera a la mejor, la CONFIRMA en un grupo B distinto: solo se acepta si gana en los dos.
    Así evitamos quedarnos con combinaciones que solo tuvieron suerte.
  - Cada mejora confirmada se guarda en starter/best_weights.json. recommender_v1.py lo relee
    cada 2 minutos y la empieza a usar en vivo sin reiniciar (con un model_version nuevo).
  - Todas las pruebas quedan en starter/tuning_log.csv (útil para la presentación).
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
BEST_FILE = HERE / "best_weights.json"
LOG_FILE = HERE / "tuning_log.csv"
MINUTES = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--minutes=")), 60))
N_EACH = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 50000))
MIN_GAIN = 0.0005   # mejora mínima en SNIPS_buy para considerarla real

GRID = {"buy_after_view": [0, 0.5, 1, 2, 3, 6, 10], "carousel": [0, 0.5, 1, 2, 4, 8],
        "coview": [0, 0.15, 0.3, 0.6, 1], "live_cooc": [0, 0.5, 1.5, 3], "self": [0, 0.4, 0.8, 1.5, 3, 6],
        "subcat": [0, 0.15, 0.3, 0.6, 1.2], "copurchase": [0.6, 1.2, 2.4, 5, 10, 20],
        "trend": [0, 0.25, 0.5, 1, 2], "quality": [0, 0.15, 0.3, 0.7, 1.5], "pop": [0, 0.5, 1, 2, 4, 8, 16]}


def load_split(cut: str, part: int) -> list[dict]:
    """Dos grupos disjuntos del periodo de test: part 0 = A (búsqueda), part 1 = B (confirmación)."""
    return v1.query(f"""
        SELECT user_id::String AS user_id, context.page::String AS page, recent_views, shown, propensity,
               ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased
        FROM shop.history_sessions
        WHERE ts >= '{cut}' AND cityHash64(request_id::String) % 2 = {part}
        ORDER BY cityHash64(request_id::String, 19) LIMIT {N_EACH}""", 600)


def score(model, test, w) -> dict:
    old = dict(v1.W)
    v1.W.update(w)
    try:
        return evaluate(lambda r: model.recommend(r["recent_views"], r["user_id"], r["page"]), test)
    finally:
        v1.W.clear()
        v1.W.update(old)


def mutate(best: dict, rng: random.Random) -> dict:
    w = dict(best)
    for k in rng.sample(list(GRID), rng.randint(1, 3)):
        r = rng.random()
        if r < 0.35:
            w[k] = round(w[k] * 2 if w[k] else rng.choice(GRID[k][1:]), 3)
        elif r < 0.7:
            w[k] = round(w[k] / 2, 3)
        elif r < 0.85:
            w[k] = 0.0
        else:
            w[k] = rng.choice(GRID[k])
    return w


def main() -> None:
    rng = random.Random()
    last_day = v1.query("SELECT toString(toDate(max(ts::DateTime64(3)))) AS d FROM shop.history_sessions", 120)[0]["d"]
    cut = str(dt.date.fromisoformat(last_day) - dt.timedelta(days=7))
    print(f"entrenamiento < {cut} <= test · {MINUTES:g} minutos")

    A, B = load_split(cut, 0), load_split(cut, 1)
    print(f"grupo A: {len(A)} visitas · grupo B: {len(B)} visitas")
    model = v1.Model()
    model.build_history(until=cut)
    catalog = v1.query("SELECT product_id::String AS product_id, subcategory::String AS subcategory, "
                       "true AS is_active FROM shop.catalog", 60)
    model.refresh_catalog(catalog_rows=catalog, out_of_stock=set())

    base_a = evaluate(lambda r: model.baseline(), A)["SNIPS_buy"]
    base_b = evaluate(lambda r: model.baseline(), B)["SNIPS_buy"]
    print(f"baseline: A={base_a}  B={base_b}")

    best = {**v1.W, **(json.loads(BEST_FILE.read_text())["W"] if BEST_FILE.exists() else {})}
    best_a, best_b = score(model, A, best)["SNIPS_buy"], score(model, B, best)["SNIPS_buy"]
    print(f"punto de partida: A={best_a}  B={best_b}  W={best}\n")

    new_log = not LOG_FILE.exists()
    log = open(LOG_FILE, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["hora", "prueba", "tipo", "SNIPS_buy_A", "SNIPS_buy_B", "aceptada", "W"])

    end = time.time() + MINUTES * 60
    i = 0
    while time.time() < end:
        i += 1
        explore = rng.random() < 0.25
        cand = {k: rng.choice(v) for k, v in GRID.items()} if explore else mutate(best, rng)
        sa = score(model, A, cand)["SNIPS_buy"]
        sb, accepted = "", False
        if sa > best_a + MIN_GAIN:
            sb = score(model, B, cand)["SNIPS_buy"]
            if sb > best_b + MIN_GAIN:
                accepted = True
                best, best_a, best_b = cand, sa, sb
                beats = best_b > base_b
                BEST_FILE.write_text(json.dumps({
                    "W": best, "SNIPS_buy_A": sa, "SNIPS_buy_B": sb, "baseline_B": base_b,
                    "beats_baseline": beats, "found_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        writer.writerow([time.strftime("%H:%M:%S"), i, "explora" if explore else "mejora", sa, sb, accepted,
                         json.dumps(cand)])
        log.flush()
        mark = "  ★ NUEVA MEJOR (guardada)" if accepted else ""
        print(f"{time.strftime('%H:%M:%S')} prueba {i:>3} [{'explora' if explore else 'mejora '}] "
              f"A={sa:.4f}{'' if sb == '' else f' B={sb:.4f}'}  mejor A={best_a:.4f} B={best_b:.4f}{mark}")

    print(f"\nFin. Mejor combinación (B={best_b} vs baseline B={base_b}):\n{json.dumps(best, indent=1)}")


if __name__ == "__main__":
    main()
