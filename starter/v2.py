"""v2 · modelo entrenado con los clientes del pasado (equipo 19)

    python starter/v2.py train     # entrena con shop.history_sessions y guarda starter/model_v2.pkl (~5-10 min)

Después, recommender_v1.py detecta model_v2.pkl al arrancar y añade la variante "v2-gbm" al bandit,
que compite en vivo con las demás.

Qué aprende: en el histórico, cada visita al carrusel mostró 10 productos y sabemos cuál se compró
(si alguno). Para cada par (visita, producto mostrado) calculamos ~25 características y entrenamos un
modelo de gradient boosting que predice la probabilidad de compra:

  relación con lo que el cliente está viendo  vista→compra, carrusel→compra, co-visitas, es lo mismo que vio,
                                               misma subcategoría / categoría que lo visto
  relación con lo que el cliente compró antes co-compras, ya lo compró, afinidad con su categoría/subcategoría
                                               (solo compras ANTERIORES a la visita: sin mirar el futuro)
  producto                                     popularidad, conversión, precio, precio relativo a lo que mira
                                               y a lo que suele comprar, categoría
  contexto                                     página, dispositivo, nº de vistas, nº de compras, posición

Las compras del histórico las decidió el mismo simulador que decide las de ahora: el modelo aprende a
imitar sus preferencias. En vivo, para cada cliente tomamos ~150 candidatos (los mejores según v1, best
sellers y lo que ha visto), el modelo puntúa cada uno y mostramos los 10 con más probabilidad de compra.
"""
from __future__ import annotations

import datetime as dt
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_main = sys.modules.get("__main__")
if _main is not None and hasattr(_main, "Model") and hasattr(_main, "VARIANTS"):
    v1 = _main                      # importado desde recommender_v1.py en marcha: misma instancia del módulo
else:
    sys.path.insert(0, str(Path(__file__).parent))
    import recommender_v1 as v1     # noqa: E402

HERE = Path(__file__).parent
MODEL_FILE = HERE / "model_v2.pkl"
PAGES = ["home", "product", "category"]
DEVICES = ["web", "ios", "android"]
FEATURES = [
    "posicion", "vista_compra", "carrusel_compra", "covisitas", "cocompras", "es_lo_visto_reciente",
    "es_lo_visto", "misma_subcat_ultima_vista", "frac_vistas_misma_subcat", "frac_vistas_misma_cat",
    "ya_comprado", "frac_compras_misma_cat", "frac_compras_misma_subcat", "popularidad", "calidad_carrusel",
    "conversion_organica", "log_precio", "precio_vs_visto", "precio_vs_comprado", "n_vistas", "n_compras",
    "pagina", "dispositivo", "categoria",
]
CATEGORICAL = ["pagina", "dispositivo", "categoria"]
N_CANDIDATES = 120


class Features:
    """Calcula las características de (contexto del cliente, producto). Igual en entrenamiento y en vivo."""

    def __init__(self, m, cats: list[str] | None = None) -> None:
        self.m = m
        self.idx = {name: {p: dict(lst) for p, lst in getattr(m, name).items()}
                    for name in ("buy_after_view", "carousel", "coview", "copurchase")}
        self.cats = cats or sorted({c for c in m.cat_of.values() if c})
        self.cat_code = {c: i + 1 for i, c in enumerate(self.cats)}     # 0 = desconocida
        self.user_hist: dict[str, list[tuple[str, str]]] = {}

    def load_user_history(self) -> None:
        """Todas las compras orgánicas con su fecha, para usar solo las anteriores a cada visita."""
        rows = v1.safe_query("historial de compras con fecha", """
SELECT user_id::String AS u, product_id::String AS p, toString(min(ts::DateTime64(3))) AS t
FROM shop.history_events WHERE event_type = 'purchase' GROUP BY u, p""")
        hist: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            hist.setdefault(r["u"], []).append((r["t"], r["p"]))
        self.user_hist = hist

    def context(self, recent_views, user_id: str, page: str, device: str, before_ts: str | None = None) -> dict:
        m = self.m
        views = [v["product_id"] for v in (recent_views or []) if isinstance(v, dict) and v.get("product_id")]
        order: list[str] = []
        for v in reversed(views):                      # más reciente primero, sin repetir
            if v not in order:
                order.append(v)
        hist = self.user_hist.get(user_id, ())
        if before_ts is not None:                      # entrenamiento: solo el pasado de esa visita
            bought = {p for t, p in hist if t < before_ts}
        else:                                          # en vivo: todo el histórico + compras de hoy
            bought = {p for _, p in hist} | m.live_bought.get(user_id, set())
        blist = list(bought)[:30]
        vprices = [m.price_of[v] for v in order if m.price_of.get(v)]
        bprices = [m.price_of[p] for p in blist if m.price_of.get(p)]
        return {
            "order": order, "w": [v1.DECAY ** k for k in range(len(order))], "viewset": set(order),
            "last_sub": m.subcat_of.get(order[0], "") if order else None,
            "vsub": Counter(m.subcat_of.get(v, "") for v in order), "vcat": Counter(m.cat_of.get(v, "") for v in order),
            "bought": bought, "blist": blist,
            "bsub": Counter(m.subcat_of.get(p, "") for p in blist), "bcat": Counter(m.cat_of.get(p, "") for p in blist),
            "vprice": sum(vprices) / len(vprices) if vprices else 0.0,
            "bprice": sum(bprices) / len(bprices) if bprices else 0.0,
            "n_views": len(views), "page": PAGES.index(page) + 1 if page in PAGES else 0,
            "device": DEVICES.index(device) + 1 if device in DEVICES else 0,
        }

    def row(self, c: dict, q: str, pos: int) -> list[float]:
        m, idx = self.m, self.idx
        bav = car = cov = selfw = 0.0
        for v, w in zip(c["order"], c["w"]):
            bav += w * idx["buy_after_view"].get(v, {}).get(q, 0.0)
            car += w * idx["carousel"].get(v, {}).get(q, 0.0)
            cov += w * idx["coview"].get(v, {}).get(q, 0.0)
            if v == q:
                selfw += w
        nb = len(c["blist"])
        cp = sum(idx["copurchase"].get(u, {}).get(q, 0.0) for u in c["blist"]) / math.sqrt(max(1, nb))
        sub, cat = m.subcat_of.get(q, ""), m.cat_of.get(q, "")
        price = m.price_of.get(q, 0.0)
        buys, views = m.sales.get(q, (0.0, 0.0))
        nv = max(1, len(c["order"]))
        return [
            pos, bav, car, cov, cp, selfw, float(q in c["viewset"]),
            float(c["last_sub"] is not None and c["last_sub"] == sub),
            c["vsub"].get(sub, 0) / nv, c["vcat"].get(cat, 0) / nv,
            float(q in c["bought"]), c["bcat"].get(cat, 0) / max(1, nb), c["bsub"].get(sub, 0) / max(1, nb),
            m.pop.get(q, 0.0), m.quality.get(q, 0.0), (buys + 1) / (views + 50), math.log1p(price),
            math.log(price / c["vprice"]) if price > 0 and c["vprice"] > 0 else 0.0,
            math.log(price / c["bprice"]) if price > 0 and c["bprice"] > 0 else 0.0,
            c["n_views"], nb, c["page"], c["device"], self.cat_code.get(cat, 0),
        ]


# ---------------------------------------------------------------- en vivo
class V2Live:
    def __init__(self, m) -> None:
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)
        self.clf = saved["clf"]
        self.m = m
        self.F = Features(m, saved["cats"])
        self.F.load_user_history()
        self.info = saved.get("metrics", {})
        print(f"  v2 cargada (entrenada {saved.get('trained_at', '?')}): {self.info}")

    def recommend(self, req: dict) -> list[str]:
        m = self.m
        c = self.F.context(req.get("recent_views"), req.get("user_id", ""), req.get("page", ""), req.get("device", ""))
        sc = m.scores(req.get("recent_views"), req.get("user_id", ""), v1.W_FIXED)
        cands = [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])[:N_CANDIDATES]]
        cands += m.best_sellers[:60] + c["order"]
        seen, pool = set(), []
        for p in cands:
            if p not in seen and p in m.available:
                seen.add(p)
                pool.append(p)
        if len(pool) < 10:
            pool += [p for p in m.popular if p not in seen][: 10 - len(pool)]
        X = np.asarray([self.F.row(c, q, 0) for q in pool], dtype=np.float32)
        prob = self.clf.predict_proba(X)[:, 1]
        order = np.argsort(-prob)[:10]
        return [pool[i] for i in order]


# ---------------------------------------------------------------- entrenamiento
def load_sessions(where: str, limit: int, chunks: int = 8) -> list[dict]:
    """Descarga en trozos (por hash del request_id) para no pedir respuestas enormes de golpe."""
    out: list[dict] = []
    for i in range(chunks):
        for attempt in range(3):
            try:
                out += v1.query(f"""
SELECT user_id::String AS user_id, toString(ts::DateTime64(3)) AS ts, context.page::String AS page,
       context.device::String AS device, recent_views, shown, ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE {where} AND cityHash64(request_id::String) % {chunks} = {i}
LIMIT {limit // chunks}""", 900)
                break
            except Exception as e:
                print(f"    trozo {i + 1}/{chunks} falló ({e}); reintento {attempt + 1}/3")
                time.sleep(5)
    return out


def build_matrix(F: Features, sessions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X, y, groups = [], [], []
    t = time.time()
    for i, s in enumerate(sessions):
        c = F.context(s["recent_views"], s["user_id"], s["page"], s["device"], before_ts=s["ts"])
        for pos, q in enumerate(s["shown"][:10]):
            X.append(F.row(c, q, pos))
            y.append(1 if s["purchased"] == q else 0)
            groups.append(i)
        if (i + 1) % 20000 == 0:
            print(f"    {i + 1}/{len(sessions)} visitas procesadas ({time.time() - t:.0f}s)")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), groups


def ranking_metrics(name: str, sessions: list[dict], scorer) -> dict:
    """En las visitas con compra: ¿en qué puesto coloca cada método al producto comprado, entre los 10 mostrados?"""
    rr, h1, h3 = [], 0, 0
    for s in sessions:
        shown = s["shown"][:10]
        if s["purchased"] not in shown:
            continue
        scores = scorer(s, shown)
        rank = 1 + sum(1 for sc, q in zip(scores, shown) if sc > scores[shown.index(s["purchased"])])
        rr.append(1 / rank)
        h1 += rank == 1
        h3 += rank <= 3
    n = max(1, len(rr))
    res = {"MRR": round(sum(rr) / n, 4), "acierto@1": round(h1 / n, 4), "acierto@3": round(h3 / n, 4), "n": len(rr)}
    print(f"  {name:<28} MRR={res['MRR']:.4f}  acierto@1={res['acierto@1']:.4f}  acierto@3={res['acierto@3']:.4f}")
    return res


def train() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    last_day = v1.query("SELECT toString(toDate(max(ts::DateTime64(3)))) AS d FROM shop.history_sessions", 120)[0]["d"]
    cut = str(dt.date.fromisoformat(last_day) - dt.timedelta(days=7))
    print(f"Entrenamiento: visitas antes de {cut} · validación: desde {cut}")

    print("1/5 Señales del histórico (solo datos anteriores al corte)...")
    m = v1.Model()
    m.build_history(until=cut)
    m.refresh_catalog(catalog_rows=v1.query(v1.SQL_CATALOG, 60), out_of_stock=set())
    F = Features(m)
    F.load_user_history()

    print("2/5 Descargando visitas del histórico...")
    pos_tr = load_sessions(f"ts < '{cut}' AND purchased IS NOT NULL AND purchased::String != ''", 60000)
    neg_tr = load_sessions(f"ts < '{cut}' AND purchased IS NULL", 120000)
    val = load_sessions(f"ts >= '{cut}' AND purchased IS NOT NULL AND purchased::String != ''", 20000)
    val_neg = load_sessions(f"ts >= '{cut}' AND purchased IS NULL", 30000)
    print(f"  entrenamiento: {len(pos_tr)} visitas con compra + {len(neg_tr)} sin compra · "
          f"validación: {len(val)} con compra + {len(val_neg)} sin compra")

    print("3/5 Calculando características...")
    X, y, _ = build_matrix(F, pos_tr + neg_tr)
    Xv, yv, _ = build_matrix(F, val + val_neg)
    print(f"  {X.shape[0]} ejemplos de entrenamiento ({int(y.sum())} compras), {X.shape[1]} características")

    print("4/5 Entrenando (gradient boosting)...")
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=80,
                                         l2_regularization=1.0, categorical_features=cat_mask,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                                         random_state=19)
    clf.fit(X, y)
    auc = roc_auc_score(yv, clf.predict_proba(Xv)[:, 1])
    print(f"  {clf.n_iter_} árboles · AUC en validación: {auc:.4f} (0.5 = azar, 1 = perfecto)")

    print("5/5 Validación: ¿quién ordena mejor los 10 productos mostrados? (visitas con compra, después del corte)")
    def v2_scorer(s, shown):
        c = F.context(s["recent_views"], s["user_id"], s["page"], s["device"], before_ts=s["ts"])
        return list(clf.predict_proba(np.asarray([F.row(c, q, 0) for q in shown], dtype=np.float32))[:, 1])

    def v1_scorer(s, shown):
        sc = m.scores(s["recent_views"], s["user_id"], v1.W_FIXED)
        return [sc.get(q, 0.0) for q in shown]

    metrics = {
        "azar (referencia)": ranking_metrics("azar (referencia)", val, lambda s, sh: list(np.random.rand(len(sh)))),
        "baseline (best sellers)": ranking_metrics("baseline (best sellers)", val,
                                                   lambda s, sh: [m.pop.get(q, 0.0) for q in sh]),
        "v1 (reglas)": ranking_metrics("v1 (reglas)", val, v1_scorer),
        "v2 (modelo)": ranking_metrics("v2 (modelo)", val, v2_scorer),
    }

    # importancia de cada característica: cuánto empeora el AUC al desordenarla
    print("\nImportancia de cada característica (pérdida de AUC al desordenarla; más = más importante):")
    rng = np.random.default_rng(19)
    sub = rng.choice(len(Xv), size=min(len(Xv), 150000), replace=False)
    Xs, ys = Xv[sub], yv[sub]
    base_auc = roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])
    imp = {}
    for j, f in enumerate(FEATURES):
        Xp = Xs.copy()
        rng.shuffle(Xp[:, j])
        imp[f] = round(base_auc - roc_auc_score(ys, clf.predict_proba(Xp)[:, 1]), 4)
    for f, d in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"  {f:<28} {d:+.4f}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"clf": clf, "features": FEATURES, "cats": F.cats, "cut": cut,
                     "trained_at": time.strftime("%Y-%m-%d %H:%M"),
                     "metrics": {"AUC": round(auc, 4), "MRR_v2": metrics["v2 (modelo)"]["MRR"],
                                 "MRR_v1": metrics["v1 (reglas)"]["MRR"],
                                 "MRR_baseline": metrics["baseline (best sellers)"]["MRR"]},
                     "importance": imp}, f)
    print(f"\nGuardado en {MODEL_FILE} ({time.time() - t0:.0f}s en total). "
          f"Reinicia recommender_v1.py para que entre en el bandit como 'v2-gbm'.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print(__doc__)
