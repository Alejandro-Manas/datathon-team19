"""v2 · IA entrenada con los clientes del pasado y los de hoy (equipo 19)

    python starter/v2.py train     # entrena y guarda starter/model_v2.pkl (copia del anterior: model_v2_prev.pkl)

Después, recommender_v1.py carga model_v2.pkl al arrancar y añade la variante "v2-gbm" al bandit.

Qué aprende: cada visita al carrusel mostró 10 productos y sabemos cuál se compró (si alguno). Para cada
par (visita, producto mostrado) se calculan ~29 características y un modelo de gradient boosting aprende
la probabilidad de compra. Las compras las decide el mismo simulador que puntúa: la IA aprende a imitarlo.

Datos (versión 2):
  1. Histórico (shop.history_sessions): visitas CON y SIN compra.
  2. Clientes de hoy: team_19.feedback (qué les mostramos y qué compraron) cruzado con shop.requests
     (quiénes eran y qué habían visto). Es el dato más valioso: clientes actuales y nuestras propias listas.
     Pesan más en el entrenamiento (LIVE_WEIGHT).
  Sin fugas de información: las señales (co-compras, carrusel→compra...) se calculan con el histórico
  ANTERIOR a las visitas de entrenamiento, para que una visita no "vea" su propia compra.

Validación: con las visitas más recientes de HOY (lo más parecido a lo que puntúa el domingo) y con la
última semana del histórico. Se compara con azar, baseline, electropapis v1 y la IA anterior.

Características:
  relación con lo que está viendo   vista→compra, carrusel→compra, co-visitas, es lo que vio, misma subcat/cat
  su historial                      co-compras, ya lo compró, afinidad con su categoría/subcategoría
                                    (solo compras ANTERIORES a la visita)
  producto                          popularidad, conversión, precio, precio relativo, categoría
  cold start y tienda de hoy        producto nuevo, compras en las últimas 24 h, cambio de precio
  contexto                          tipo de cliente, página, dispositivo, nº de vistas/compras, posición,
                                    si el dato es de hoy
"""
from __future__ import annotations

import datetime as dt
import math
import pickle
import shutil
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
VERSION = 2
PAGES = ["home", "product", "category"]
DEVICES = ["web", "ios", "android"]
TYPES = ["nuevo", "prod_nuevo", "sin_vistas", "producto", "categoria"]
FEATURES = [
    "posicion", "vista_compra", "carrusel_compra", "covisitas", "cocompras", "es_lo_visto_reciente",
    "es_lo_visto", "misma_subcat_ultima_vista", "frac_vistas_misma_subcat", "frac_vistas_misma_cat",
    "ya_comprado", "frac_compras_misma_cat", "frac_compras_misma_subcat", "popularidad", "calidad_carrusel",
    "conversion_organica", "log_precio", "precio_vs_visto", "precio_vs_comprado", "n_vistas", "n_compras",
    "pagina", "dispositivo", "categoria",
    # nuevas en la versión 2
    "producto_nuevo", "compras_24h", "cambio_precio", "tipo_cliente", "dato_de_hoy",
]
CATEGORICAL = ["pagina", "dispositivo", "categoria", "tipo_cliente"]
N_CANDIDATES = 120
LIVE_WEIGHT = 3.0          # cuánto más pesa una visita de hoy que una del histórico al entrenar
SIGNAL_DAYS = 14           # días de histórico reservados para entrenar (las señales usan lo anterior)


class Features:
    """Calcula las características de (contexto del cliente, producto). Igual en entrenamiento y en vivo."""

    def __init__(self, m, cats: list[str] | None = None) -> None:
        self.m = m
        self.idx = {name: {p: dict(lst) for p, lst in getattr(m, name).items()}
                    for name in ("buy_after_view", "carousel", "coview", "copurchase")}
        self.cats = cats or sorted({c for c in m.cat_of.values() if c})
        self.cat_code = {c: i + 1 for i, c in enumerate(self.cats)}     # 0 = desconocida
        self.user_hist: dict[str, list[tuple[str, str]]] = {}
        self.base_price: dict[str, float] = {}

    def load(self) -> None:
        """Compras de cada cliente con su fecha (para usar solo las anteriores a cada visita) y precios de partida."""
        rows = v1.safe_query("historial de compras con fecha", """
SELECT user_id::String AS u, product_id::String AS p, toString(min(ts::DateTime64(3))) AS t
FROM shop.history_events WHERE event_type = 'purchase' GROUP BY u, p""")
        hist: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            hist.setdefault(r["u"], []).append((r["t"], r["p"]))
        self.user_hist = hist
        self.base_price = {r["product_id"]: float(r["price"] or 0) for r in v1.safe_query("precios de partida", """
SELECT product_id::String AS product_id, price_eur::Float64 AS price FROM shop.catalog""")}

    def context(self, recent_views, user_id: str, page: str, device: str, before_ts: str | None = None,
                is_new: bool | None = None, live: bool = True) -> dict:
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
        # tipo de cliente (mismas reglas que recommender_v1.segment; en el histórico "nuevo" = sin compras previas)
        new_user = bool(is_new) if is_new is not None else not bought
        if new_user:
            typ = "nuevo"
        elif any(v in m.new_products for v in order):
            typ = "prod_nuevo"
        elif not order:
            typ = "sin_vistas"
        else:
            typ = "categoria" if page == "category" else "producto"
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
            "type": TYPES.index(typ) + 1, "live": 1.0 if live else 0.0,
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
        base = self.base_price.get(q, 0.0)
        buys, views = m.sales.get(q, (0.0, 0.0))
        nv = max(1, len(c["order"]))
        live = c["live"]
        return [
            pos, bav, car, cov, cp, selfw, float(q in c["viewset"]),
            float(c["last_sub"] is not None and c["last_sub"] == sub),
            c["vsub"].get(sub, 0) / nv, c["vcat"].get(cat, 0) / nv,
            float(q in c["bought"]), c["bcat"].get(cat, 0) / max(1, nb), c["bsub"].get(sub, 0) / max(1, nb),
            m.pop.get(q, 0.0), m.quality.get(q, 0.0), (buys + 1) / (views + 50), math.log1p(price),
            math.log(price / c["vprice"]) if price > 0 and c["vprice"] > 0 else 0.0,
            math.log(price / c["bprice"]) if price > 0 and c["bprice"] > 0 else 0.0,
            c["n_views"], nb, c["page"], c["device"], self.cat_code.get(cat, 0),
            # cold start y tienda de hoy (en las visitas del histórico no existen: 0)
            float(q in m.new_products) * live, m.pop_24h.get(q, 0.0) * live,
            (math.log(price / base) if price > 0 and base > 0 else 0.0) * live,
            c["type"], live,
        ]


# ---------------------------------------------------------------- en vivo
class V2Live:
    def __init__(self, m) -> None:
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)
        if saved.get("features") != FEATURES:
            raise RuntimeError("model_v2.pkl es de otra versión del código: vuelve a entrenar con "
                               "'python starter/v2.py train'")
        self.clf = saved["clf"]
        self.m = m
        self.F = Features(m, saved["cats"])
        self.F.load()
        self.info = saved.get("metrics", {})
        print(f"  v2 cargada (entrenada {saved.get('trained_at', '?')}): {self.info}")

    def recommend(self, req: dict) -> list[str]:
        m = self.m
        c = self.F.context(req.get("recent_views"), req.get("user_id", ""), req.get("page", ""),
                           req.get("device", ""), is_new=req.get("is_new_user") in (True, 1, "true"), live=True)
        sc = m.scores(req.get("recent_views"), req.get("user_id", ""), v1.EP)
        cands = [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])[:N_CANDIDATES]]
        cands += m.best_sellers[:60] + c["order"]
        cands += sorted(m.pop_24h, key=lambda p: -m.pop_24h[p])[:30]          # lo que se vende hoy
        cands += [p for p in m.new_products if p in m.available][:30]          # productos nuevos (cold start)
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


# ---------------------------------------------------------------- datos de entrenamiento
def _chunked(sql_for_chunk, chunks: int = 8) -> list[dict]:
    out: list[dict] = []
    for i in range(chunks):
        for attempt in range(3):
            try:
                out += v1.query(sql_for_chunk(i, chunks), 900)
                break
            except Exception as e:
                print(f"    trozo {i + 1}/{chunks} falló ({e}); reintento {attempt + 1}/3")
                time.sleep(5)
    return out


def load_history(where: str, limit: int) -> list[dict]:
    rows = _chunked(lambda i, k: f"""
SELECT user_id::String AS user_id, toString(ts::DateTime64(3)) AS ts, context.page::String AS page,
       context.device::String AS device, recent_views, shown, ifNull(purchased::String, '') AS purchased
FROM shop.history_sessions
WHERE {where} AND cityHash64(request_id::String) % {k} = {i}
LIMIT {limit // k}""")
    for r in rows:
        r["live"] = False
    return rows


def load_live() -> list[dict]:
    """Clientes de hoy: lo que les mostramos y lo que compraron (feedback) + quiénes eran (requests)."""
    rows = _chunked(lambda i, k: f"""
SELECT r.uid AS user_id, r.t AS ts, r.pg AS page, r.dev AS device, r.nu AS is_new_user,
       r.rv AS recent_views, f.shown AS shown, f.pur AS purchased
FROM (SELECT request_id::String AS rid, shown, ifNull(purchased::String, '') AS pur FROM {v1.TEAM}.feedback
      WHERE request_id::String != '__init__' AND cityHash64(request_id::String) % {k} = {i}) AS f
INNER JOIN (SELECT request_id::String AS rid, user_id::String AS uid, toString(ts::DateTime64(3)) AS t,
                   context.page::String AS pg, context.device::String AS dev, is_new_user AS nu, recent_views AS rv
            FROM shop.requests WHERE cityHash64(request_id::String) % {k} = {i}) AS r ON f.rid = r.rid""")
    out = []
    for r in rows:
        if isinstance(r.get("shown"), list) and len(r["shown"]) >= 10:
            r["live"] = True
            out.append(r)
    out.sort(key=lambda r: r["ts"])
    return out


def build_matrix(F: Features, sessions: list[dict]):
    X, y, w = [], [], []
    t = time.time()
    for i, s in enumerate(sessions):
        live = s.get("live", False)
        c = F.context(s["recent_views"], s["user_id"], s["page"], s["device"], before_ts=s["ts"],
                      is_new=(s.get("is_new_user") in (True, 1, "true")) if live else None, live=live)
        for pos, q in enumerate(s["shown"][:10]):
            X.append(F.row(c, q, pos))
            y.append(1 if s["purchased"] == q else 0)
            w.append(LIVE_WEIGHT if live else 1.0)
        if (i + 1) % 25000 == 0:
            print(f"    {i + 1}/{len(sessions)} visitas procesadas ({time.time() - t:.0f}s)")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(w, dtype=np.float32)


def ranking_metrics(name: str, sessions: list[dict], scorer) -> dict:
    """En las visitas con compra: ¿en qué puesto coloca cada método al producto comprado, entre los 10 mostrados?"""
    rr, h1, h3 = [], 0, 0
    for s in sessions:
        shown = s["shown"][:10]
        if s["purchased"] not in shown:
            continue
        scores = scorer(s, shown)
        target = scores[shown.index(s["purchased"])]
        rank = 1 + sum(1 for sc in scores if sc > target)
        rr.append(1 / rank)
        h1 += rank == 1
        h3 += rank <= 3
    n = max(1, len(rr))
    res = {"MRR": round(sum(rr) / n, 4), "acierto@1": round(h1 / n, 4), "acierto@3": round(h3 / n, 4), "n": len(rr)}
    print(f"  {name:<26} MRR={res['MRR']:.4f}  acierto@1={res['acierto@1']:.4f}  acierto@3={res['acierto@3']:.4f}"
          f"  ({len(rr)} compras)")
    return res


def train() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    last_day = v1.query("SELECT toString(toDate(max(ts::DateTime64(3)))) AS d FROM shop.history_sessions", 120)[0]["d"]
    cut = str(dt.date.fromisoformat(last_day) - dt.timedelta(days=7))            # validación histórica desde aquí
    sig_cut = str(dt.date.fromisoformat(cut) - dt.timedelta(days=SIGNAL_DAYS))   # señales con datos anteriores
    print(f"Señales: histórico < {sig_cut} · entrenamiento: histórico {sig_cut}..{cut} + clientes de hoy · "
          f"validación: histórico >= {cut} + los clientes más recientes de hoy")

    print("1/6 Señales del histórico (solo datos anteriores a las visitas de entrenamiento)...")
    m = v1.Model()
    m.build_history(until=sig_cut)
    m.refresh_catalog(catalog_rows=v1.query(v1.SQL_CATALOG, 60), out_of_stock=set())
    try:
        m.refresh_trend()
    except Exception as e:
        print(f"  aviso: sin compras de las últimas 24 h ({e})")
    F = Features(m)
    F.load()

    print("2/6 Descargando visitas del histórico (con y sin compra)...")
    hist_pos = load_history(f"ts >= '{sig_cut}' AND ts < '{cut}' AND ifNull(purchased::String, '') != ''", 60000)
    hist_neg = load_history(f"ts >= '{sig_cut}' AND ts < '{cut}' AND ifNull(purchased::String, '') = ''", 100000)
    val_hist = load_history(f"ts >= '{cut}' AND ifNull(purchased::String, '') != ''", 20000)
    val_hist_neg = load_history(f"ts >= '{cut}' AND ifNull(purchased::String, '') = ''", 30000)
    print(f"  histórico: {len(hist_pos)} con compra + {len(hist_neg)} sin compra (entrenamiento) · "
          f"{len(val_hist)} + {len(val_hist_neg)} (validación)")

    print("3/6 Descargando clientes de hoy (feedback × requests)...")
    live = load_live()
    n_val = len(live) // 5                       # el 20 % más reciente, para validar
    live_tr, live_val = live[:-n_val] if n_val else live, live[-n_val:] if n_val else []
    print(f"  hoy: {len(live)} clientes ({sum(1 for r in live if r['purchased'])} con compra) · "
          f"entrenamiento {len(live_tr)} · validación {len(live_val)} (los más recientes)")

    print("4/6 Calculando características...")
    X, y, w = build_matrix(F, hist_pos + hist_neg + live_tr)
    Xv, yv, _ = build_matrix(F, val_hist + val_hist_neg + live_val)
    print(f"  {X.shape[0]} ejemplos de entrenamiento ({int(y.sum())} compras), {X.shape[1]} características")

    print("5/6 Entrenando (gradient boosting)...")
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=80,
                                         l2_regularization=1.0,
                                         categorical_features=[f in CATEGORICAL for f in FEATURES],
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                         random_state=19)
    clf.fit(X, y, sample_weight=w)
    auc = roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]) if 0 < yv.sum() < len(yv) else float("nan")
    print(f"  {clf.n_iter_} árboles · AUC en validación: {auc:.4f} (0.5 = azar, 1 = perfecto)")

    print("6/6 Validación: ¿quién ordena mejor los 10 productos mostrados?")
    def scorer_v2(model, feats):
        def sc(s, shown):
            lv = s.get("live", False)
            c = feats.context(s["recent_views"], s["user_id"], s["page"], s["device"], before_ts=s["ts"],
                              is_new=(s.get("is_new_user") in (True, 1, "true")) if lv else None, live=lv)
            return list(model.predict_proba(np.asarray([feats.row(c, q, 0) for q in shown], dtype=np.float32))[:, 1])
        return sc

    def ep_scorer(s, shown):
        sc = m.scores(s["recent_views"], s["user_id"], v1.EP)
        return [sc.get(q, 0.0) for q in shown]

    metrics = {}
    for label, val in (("CLIENTES DE HOY (lo que puntúa)", live_val), ("última semana del histórico", val_hist)):
        if not val:
            continue
        print(f" {label}:")
        res = {
            "azar": ranking_metrics("azar (referencia)", val, lambda s, sh: list(np.random.rand(len(sh)))),
            "baseline": ranking_metrics("baseline (best sellers)", val, lambda s, sh: [m.pop.get(q, 0.0) for q in sh]),
            "electropapis v1": ranking_metrics("electropapis v1 (reglas)", val, ep_scorer),
            "v2 nueva": ranking_metrics("IA nueva", val, scorer_v2(clf, F)),
        }
        metrics[label] = res

    print("\nImportancia de cada característica (pérdida de AUC al desordenarla; más = más importante):")
    rng = np.random.default_rng(19)
    sub = rng.choice(len(Xv), size=min(len(Xv), 150000), replace=False)
    Xs, ys = Xv[sub], yv[sub]
    base_auc = roc_auc_score(ys, clf.predict_proba(Xs)[:, 1]) if 0 < ys.sum() < len(ys) else 0.0
    imp = {}
    for j, f in enumerate(FEATURES):
        Xp = Xs.copy()
        rng.shuffle(Xp[:, j])
        imp[f] = round(base_auc - (roc_auc_score(ys, clf.predict_proba(Xp)[:, 1]) if base_auc else 0.0), 4)
    for f, d in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"  {f:<28} {d:+.4f}")

    if MODEL_FILE.exists():
        shutil.copy(MODEL_FILE, HERE / "model_v2_prev.pkl")
        print(f"\nCopia de la IA anterior en {HERE / 'model_v2_prev.pkl'}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"clf": clf, "features": FEATURES, "cats": F.cats, "version": VERSION,
                     "trained_at": time.strftime("%Y-%m-%d %H:%M"), "metrics": {"AUC": round(float(auc), 4)},
                     "validation": metrics, "importance": imp},
                    f)
    print(f"Guardado en {MODEL_FILE} ({time.time() - t0:.0f}s en total). "
          f"Reinicia recommender_v1.py para que entre en el bandit como 'v2-gbm'.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print(__doc__)
