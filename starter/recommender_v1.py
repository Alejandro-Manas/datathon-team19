"""Recomendador v1 · equipo 19

    python starter/recommender_v1.py        # sustituye a recommender.py (¡no ejecutar los dos a la vez!)

Idea: para cada comprador puntuamos productos candidatos combinando varias señales
calculadas en SQL dentro de RawTree y las ordenamos:

  1. Vistas de la sesión → compras   (shop.history_events: "quien vio A compró B en la misma sesión")
  2. Carrusel antiguo → compras       (shop.history_sessions: "con A en recent_views, se compró B del carrusel")
  3. Co-visitas                        ("quien vio A también vio B")
  4. Mismo subcategoría que lo visto  (cold start: productos nuevos sin historial)
  5. Historial del usuario             (co-compras de lo que ya compró; penalizamos repetir compra)
  6. Tendencia en vivo                 (compras/carritos de la última hora en shop.events)
  7. Calidad del producto              (tasa de compra cuando el carrusel lo mostró)

Las vistas más recientes pesan más (decaimiento 0.6^k). Un hilo en segundo plano refresca
stock, tendencias y co-visitas en vivo sin bloquear el bucle de respuesta.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_19")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

MODEL_VERSION = "v1"   # prefijo; cada variante del bandit tiene su propio nombre
Record = dict[str, Any]

# Autoajuste en vivo POR TIPO DE CLIENTE (bandit contextual con muestreo de Thompson):
#   - Cada cliente se clasifica en un tipo (ver segment(): nuevo, ve un producto nuevo, habitual sin vistas,
#     habitual en ficha de producto, habitual en categoría).
#   - Para cada tipo, el bandit lleva la cuenta de compras reales de cada variante y manda más clientes de
#     ese tipo a la variante que más les vende. Así cada tipo de cliente acaba con "su" variante.
#   - SHARE_CONTROL de las peticiones va al baseline, como referencia.
#   - FINAL_MODE = True (domingo): sin control y, para cada tipo de cliente, siempre su mejor variante.
# Ver resultados: python starter/ab_report.py 3        ·   por tipo de cliente: python starter/ab_report.py 3 grupos
SHARE_CONTROL = 0.05
FINAL_MODE = False
# Si se pone el nombre de una variante, TODOS los clientes la reciben (sin control ni bandit).
# Sirve para ver su resultado real "a solas" en el leaderboard. None = funcionamiento normal (bandit).
FORCE_VARIANT = None
STATS_HOURS = 12            # ventana de feedback que usa el bandit

# Pesos de cada señal (a ajustar con eval_v1.py)
W = {
    "buy_after_view": 3.0,
    "carousel": 2.0,
    "coview": 1.0,
    "live_cooc": 1.5,
    "self": 0.8,          # volver a mostrar lo que acaba de ver
    "subcat": 0.6,
    "copurchase": 1.2,
    "trend": 0.5,
    "quality": 0.7,
    "pop": 1.0,           # best sellers de siempre (lo que hace fuerte al baseline)
}
W_FIXED = {**W, "pop": 0.0}   # la v1 original (13:05); base de todas las variantes
# electropapis v1: la mejor de la búsqueda offline (eval_v1.py --search) y la mejor en vivo (14/11)
EP = {**W_FIXED, "buy_after_view": 3, "carousel": 2, "coview": 1, "self": 2, "subcat": 1.5,
      "copurchase": 10, "quality": 0, "pop": 1}
VARIANTS = {
    "electropapis v1": EP,                                                    # la madre
    # hijas: cada una refuerza lo que debería servir más a un tipo de cliente
    "ep-historial": {**EP, "copurchase": 20, "self": 1},                      # habituales: su historial de compras
    "ep-sesion": {**EP, "self": 4, "buy_after_view": 6, "carousel": 4,
                  "copurchase": 4},                                           # en ficha: lo que está mirando ahora
    "ep-novedad": {**EP, "subcat": 3, "live_cooc": 4, "trend": 2, "pop": 2},  # nuevos / productos nuevos: tendencia y parecidos
    "ep-tendencia": {**EP, "trend": 3, "live_cooc": 3},                       # lo que se mueve hoy en la tienda
}
# Variantes especializadas por tipo de cliente, afinadas con el histórico por tune_segments.py.
# "ep-seg-<tipo>" solo compite con los clientes de ese tipo. Se cargan al arrancar si existe el archivo.
SEGMENT_FILE = Path(__file__).parent / "segment_weights.json"
SEG_VARIANT_FOR = {"prod_nuevo": "nuevo"}   # en el histórico no hay productos nuevos: usan la de "nuevo"
# Solo los tipos con cientos de compras en el histórico (ficha ~1.000, categoría ~440). "nuevo" (~12 compras) y
# "sin_vistas" (~22) no dan para afinar con fiabilidad: ahí deciden las variantes generales y el bandit en vivo.
TRUSTED_SEGMENTS = {"producto", "categoria"}
if SEGMENT_FILE.exists():
    try:
        for _sg, _d in json.loads(SEGMENT_FILE.read_text())["segments"].items():
            if _d.get("mejora") and _sg in TRUSTED_SEGMENTS:
                VARIANTS[f"ep-seg-{_sg}"] = {**EP, **{k: float(x) for k, x in _d["W"].items()}}
    except Exception as _e:
        print(f"aviso: no se pudo leer {SEGMENT_FILE.name} ({_e})")
DECAY = 0.6               # peso de la vista k-ésima más reciente: DECAY**k
ALREADY_BOUGHT = 0.15     # multiplicador para productos que el usuario ya compró
NEIGHBORS = 30


def query(sql: str, timeout: float = 10) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if r.status_code >= 400:   # mostrar el motivo que da RawTree, no solo "400 Bad Request"
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    return r.json()["data"]


def safe_query(name: str, sql: str, timeout: float = 600) -> list[Record]:
    t = time.time()
    try:
        rows = query(sql, timeout)
        print(f"  {name}: {len(rows)} filas en {time.time() - t:.1f}s")
        return rows
    except Exception as e:
        print(f"  {name}: FALLÓ ({e}); se ignora esta señal")
        return []


def neighbors(rows: list[Record]) -> dict[str, list[tuple[str, float]]]:
    """[{p, q, n}] → {p: [(q, n / max_n), ...]} ordenado por n."""
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        out[r["p"]].append((r["q"], float(r["n"])))
    for p, lst in out.items():
        lst.sort(key=lambda x: -x[1])
        top = lst[0][1] or 1.0
        out[p] = [(q, n / top) for q, n in lst]
    return dict(out)


# ---------------------------------------------------------------- SQL
def _cut(until: str | None, col: str = "ts") -> str:
    return f" AND {col} < '{until}'" if until else ""


def sql_buy_after_view(until):
    c = _cut(until)
    return f"""
SELECT v.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view'{c}) AS v
INNER JOIN (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'purchase'{c}) AS b ON v.session_id = b.session_id
GROUP BY p, q ORDER BY n DESC LIMIT {NEIGHBORS} BY p"""


def sql_coview(until):
    c = _cut(until)
    return f"""
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view'{c}) AS a
INNER JOIN (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view'{c}) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT {NEIGHBORS} BY p"""


def sql_carousel(until):
    c = _cut(until)
    return f"""
SELECT rv.product_id::String AS p, purchased::String AS q, count() AS n
FROM shop.history_sessions ARRAY JOIN recent_views AS rv
WHERE purchased IS NOT NULL AND purchased::String != ''{c}
GROUP BY p, q ORDER BY n DESC LIMIT {NEIGHBORS} BY p"""


def sql_copurchase(until):
    c = _cut(until)
    return f"""
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT DISTINCT user_id::String AS user_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'purchase'{c}) AS a
INNER JOIN (SELECT DISTINCT user_id::String AS user_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'purchase'{c}) AS b ON a.user_id = b.user_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT {NEIGHBORS} BY p"""


def sql_quality(until):
    """Tasa de compra de cada producto cuando el carrusel antiguo lo mostró (suavizada)."""
    c = _cut(until)
    return f"""
SELECT s AS product_id, count() AS n_shown, countIf(pur = s) AS bought
FROM (SELECT CAST(shown, 'Array(String)') AS sh, ifNull(purchased::String, '') AS pur
      FROM shop.history_sessions WHERE 1{c})
ARRAY JOIN sh AS s
GROUP BY s"""


def sql_sales(until):
    c = _cut(until)
    return f"""
SELECT product_id::String AS product_id, countIf(event_type = 'purchase') AS buys,
       countIf(event_type = 'view') AS views
FROM shop.history_events WHERE 1{c} GROUP BY product_id"""


def sql_user_purchases(until):
    c = _cut(until)
    return f"""
SELECT user_id::String AS u, groupUniqArray(product_id::String) AS ps
FROM shop.history_events WHERE event_type = 'purchase'{c} GROUP BY u"""


SQL_CATALOG = """
SELECT product_id, argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
       argMax(price_eur, v) AS price_eur, argMax(is_active, v) AS is_active,
       argMax(stock, v) AS stock, max(is_new) AS is_new
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS stock, 0 AS is_new, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, price_eur::Float64, is_active::Bool,
           if(change::String IN ('out_of_stock', 'back_in_stock'), change::String, ''),
           if(change::String = 'new_product', 1, 0), toString(ts::DateTime64(3))
    FROM shop.catalog_updates
)
GROUP BY product_id"""

# El stock se calcula aparte: la última actualización de stock manda, aunque luego haya un cambio de precio
SQL_OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

SQL_TREND = """
SELECT product_id::String AS product_id, countIf(event_type = 'purchase') AS buys,
       countIf(event_type = 'add_to_cart') AS carts, countIf(event_type = 'view') AS views
FROM shop.events WHERE ts::DateTime64(3) > now() - INTERVAL 1 HOUR
GROUP BY product_id"""

SQL_POP_24H = """
SELECT product_id::String AS product_id, countIf(event_type = 'purchase') AS buys
FROM shop.events WHERE ts::DateTime64(3) > now() - INTERVAL 24 HOUR
GROUP BY product_id"""

SQL_LIVE_COOC = f"""
SELECT a.product_id AS p, b.product_id AS q,
       count() + 3 * countIf(b.event_type = 'purchase') + countIf(b.event_type = 'add_to_cart') AS n
FROM (SELECT DISTINCT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.events WHERE event_type = 'view' AND ts::DateTime64(3) > now() - INTERVAL 6 HOUR) AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
      FROM shop.events WHERE ts::DateTime64(3) > now() - INTERVAL 6 HOUR) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT {NEIGHBORS} BY p"""

SQL_LIVE_PURCHASES = """
SELECT user_id::String AS u, groupUniqArray(product_id::String) AS ps
FROM shop.events WHERE event_type = 'purchase' GROUP BY u"""


# ---------------------------------------------------------------- modelo
class Model:
    def __init__(self) -> None:
        self.buy_after_view: dict = {}
        self.carousel: dict = {}
        self.coview: dict = {}
        self.copurchase: dict = {}
        self.live_cooc: dict = {}
        self.quality: dict[str, float] = {}
        self.trend: dict[str, float] = {}
        self.user_bought: dict[str, set[str]] = {}
        self.live_bought: dict[str, set[str]] = {}
        self.subcat_of: dict[str, str] = {}
        self.cat_of: dict[str, str] = {}
        self.price_of: dict[str, float] = {}
        self.new_products: set[str] = set()
        self.pop_24h: dict[str, float] = {}
        self.base_price: dict[str, float] = {}
        self.sales: dict[str, tuple[float, float]] = {}
        self.subcat_top: dict[str, list[tuple[str, float]]] = {}
        self.available: set[str] = set()
        self.popular: list[str] = []      # orden de relleno, siempre válido
        self.pop: dict[str, float] = {}
        self.best_sellers: list[str] = []

    # ---- construcción a partir del histórico (una vez al arrancar; `until` para evaluación offline)
    def build_history(self, until: str | None = None) -> None:
        print("Construyendo señales del histórico...")
        self.buy_after_view = neighbors(safe_query("vista→compra", sql_buy_after_view(until)))
        self.carousel = neighbors(safe_query("carrusel→compra", sql_carousel(until)))
        self.coview = neighbors(safe_query("co-visitas", sql_coview(until)))
        self.copurchase = neighbors(safe_query("co-compras", sql_copurchase(until)))

        q = {}
        for r in safe_query("calidad carrusel", sql_quality(until)):
            # media bayesiana: tasa global ~1.4 %, con peso de 200 impresiones
            q[r["product_id"]] = (float(r["bought"]) + 200 * 0.014) / (float(r["n_shown"]) + 200)
        sales = {r["product_id"]: (float(r["buys"]), float(r["views"]))
                 for r in safe_query("ventas", sql_sales(until))}
        for p, (b, v) in sales.items():   # productos que el carrusel nunca mostró: usamos compra/vista orgánica
            if p not in q:
                q[p] = (b + 20 * 0.05) / (v + 20) * 0.3
        top = max(q.values(), default=1.0) or 1.0
        self.quality = {p: x / top for p, x in q.items()}

        self.user_bought = {r["u"]: set(r["ps"]) for r in safe_query("compras por usuario", sql_user_purchases(until))}
        # best sellers de siempre (= baseline / fallback oficial), para el grupo de control del A/B
        self.best_sellers = [p for _, p in sorted(((b, p) for p, (b, _) in sales.items() if b > 0), reverse=True)]
        self.sales = sales
        top_b = max((b for b, _ in sales.values()), default=1.0) or 1.0
        self.pop = {p: (b / top_b) ** 0.5 for p, (b, _) in sales.items() if b > 0}

    def baseline(self) -> list[str]:
        return [p for p in self.best_sellers if p in self.available][:10]

    # ---- catálogo y stock (cada pocos segundos)
    def refresh_catalog(self, catalog_rows: list[Record] | None = None, out_of_stock: set[str] | None = None) -> None:
        rows = catalog_rows if catalog_rows is not None else query(SQL_CATALOG, 30)
        oos = out_of_stock if out_of_stock is not None else {r["product_id"] for r in query(SQL_OUT_OF_STOCK, 10)}
        available, subcat_of, cat_of, price_of = set(), {}, {}, {}
        by_sub: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            p = r["product_id"]
            subcat_of[p] = r["subcategory"]
            cat_of[p] = r.get("category", "")
            price_of[p] = float(r.get("price_eur") or 0.0)
            if r["is_active"] in (True, 1, "true") and p not in oos:
                available.add(p)
                by_sub[r["subcategory"]].append(p)
        self.subcat_of, self.cat_of, self.price_of = subcat_of, cat_of, price_of
        self.new_products = {r["product_id"] for r in rows if r.get("is_new") in (1, True, "1")}
        self.subcat_top = {s: self._rank_norm(ps) for s, ps in by_sub.items()}
        self.popular = [p for p in sorted(available, key=lambda p: -self._prior(p))]
        self.available = available

    def _prior(self, p: str) -> float:
        return self.pop.get(p, 0.0) + 0.3 * self.quality.get(p, 0.0) + 0.5 * self.trend.get(p, 0.0)

    def _rank_norm(self, ps: list[str], k: int = 15) -> list[tuple[str, float]]:
        ps = sorted(ps, key=lambda p: -self._prior(p))[:k]
        return [(p, 1.0 / (1 + i)) for i, p in enumerate(ps)]

    # ---- señales en vivo
    def refresh_trend(self) -> None:
        t = {}
        for r in query(SQL_TREND, 30):
            t[r["product_id"]] = float(r["buys"]) + 0.3 * float(r["carts"]) + 0.02 * float(r["views"])
        top = max(t.values(), default=1.0) or 1.0
        self.trend = {p: x / top for p, x in t.items()}
        d = {r["product_id"]: float(r["buys"]) for r in query(SQL_POP_24H, 30)}
        top = max(d.values(), default=1.0) or 1.0
        self.pop_24h = {p: (x / top) ** 0.5 for p, x in d.items()}   # compras en la tienda en las últimas 24 h

    def refresh_live_cooc(self) -> None:
        self.live_cooc = neighbors(query(SQL_LIVE_COOC, 120))

    def refresh_live_purchases(self) -> None:
        self.live_bought = {r["u"]: set(r["ps"]) for r in query(SQL_LIVE_PURCHASES, 60)}

    # ---- recomendación
    def recommend(self, recent_views: list[Record] | None, user_id: str = "", page: str = "",
                  w: dict | None = None) -> list[str]:
        score = self.scores(recent_views, user_id, w)
        avail = self.available
        ranked = [p for p, _ in sorted(score.items(), key=lambda x: -x[1]) if p in avail]
        out = ranked[:10]
        if len(out) < 10:   # relleno: siempre 10 productos distintos y disponibles
            have = set(out)
            out += [p for p in self.popular if p not in have][: 10 - len(out)]
        return out

    def scores(self, recent_views: list[Record] | None, user_id: str = "", w: dict | None = None) -> dict[str, float]:
        """Puntuación v1 de cada producto candidato (también la usa la v2 para elegir candidatos)."""
        W = w if w is not None else globals()["W"]
        score: dict[str, float] = defaultdict(float)
        views = [v["product_id"] for v in (recent_views or []) if isinstance(v, dict) and v.get("product_id")]

        # 1-4: vistas de la sesión, la más reciente primero
        seen = set()
        k = 0
        for v in reversed(views):
            if v in seen:
                continue
            seen.add(v)
            w = DECAY ** k
            k += 1
            for name, table in (("buy_after_view", self.buy_after_view), ("carousel", self.carousel),
                                ("coview", self.coview), ("live_cooc", self.live_cooc)):
                for q, s in table.get(v, ()):
                    score[q] += W[name] * w * s
            score[v] += W["self"] * w
            for q, s in self.subcat_top.get(self.subcat_of.get(v, ""), ()):
                score[q] += W["subcat"] * w * s

        # 5: historial del usuario
        bought = self.user_bought.get(user_id, set()) | self.live_bought.get(user_id, set())
        for u in list(bought)[:20]:
            for q, s in self.copurchase.get(u, ()):
                score[q] += W["copurchase"] * s / max(1, len(bought)) ** 0.5

        # 6-7: tendencia y calidad (también para quien no tiene historial)
        for p in self.popular[:300]:
            score[p] += W["quality"] * self.quality.get(p, 0.0) + W["trend"] * self.trend.get(p, 0.0)
        for p in self.best_sellers[:300]:
            score[p] += W["pop"] * self.pop.get(p, 0.0)

        for p in bought:
            if p in score:
                score[p] *= ALREADY_BOUGHT
        return score


# ---------------------------------------------------------------- autoajuste en vivo
PARENT = "electropapis v1"
SEGMENTS = ["nuevo", "prod_nuevo", "sin_vistas", "producto", "categoria"]
SEGMENT_NAMES = {"nuevo": "cliente nuevo", "prod_nuevo": "ve un producto nuevo", "sin_vistas": "habitual sin vistas",
                 "producto": "habitual en ficha", "categoria": "habitual en categoría"}


def segment(req: dict, model: "Model") -> str:
    """Tipo de cliente. Los dos primeros son las peticiones 'cold start', que puntúan aparte (10 puntos)."""
    if req.get("is_new_user") in (True, 1, "true"):
        return "nuevo"
    views = [v.get("product_id") for v in (req.get("recent_views") or []) if isinstance(v, dict)]
    if any(v in model.new_products for v in views):
        return "prod_nuevo"
    if not views:
        return "sin_vistas"
    return "categoria" if req.get("page") == "category" else "producto"


class Bandit:
    """Bandit contextual: para cada tipo de cliente reparte el tráfico entre VARIANTS según sus compras
    reales (muestreo de Thompson). Cada tipo empieza con lo aprendido con todos los clientes (prior
    jerárquico) y se va especializando a medida que acumula sus propias compras.
    El tipo de cada cliente se guarda en {TEAM}.scratch para poder contar compras por tipo."""

    PRIOR_STRENGTH = 300   # cuántos "clientes" pesa lo aprendido en general frente a los datos del tipo

    def __init__(self) -> None:
        self.glob: dict[str, tuple[int, int]] = {}                 # variante → (clientes, compras)
        self.seg: dict[tuple[str, str], tuple[int, int]] = {}      # (variante, tipo) → (clientes, compras)
        self.rng = random.Random()

    @property
    def stats(self) -> dict[str, tuple[int, int]]:
        return {v: self.glob.get(v, (0, 0)) for v in VARIANTS}

    def refresh(self) -> None:
        feedback = f"""
FROM (SELECT request_id::String AS rid, purchased FROM {TEAM}.feedback
      WHERE served_by::String = 'team' AND ts::DateTime64(3) > now() - INTERVAL {STATS_HOURS} HOUR) AS f
INNER JOIN (SELECT request_id::String AS rid, any(model_version::String) AS v
            FROM {TEAM}.submissions GROUP BY rid) AS s ON f.rid = s.rid"""
        try:
            rows = query(f"""
SELECT s.v AS v, ifNull(c.sg, '') AS sg, count() AS n, countIf(f.purchased::String != '') AS buys{feedback}
LEFT JOIN (SELECT request_id::String AS rid, any(segment::String) AS sg
           FROM {TEAM}.scratch WHERE kind::String = 'segment' GROUP BY rid) AS c ON f.rid = c.rid
GROUP BY v, sg""", 30)
        except Exception:   # scratch todavía vacía (primeros segundos): solo la cuenta general
            rows = query(f"""
SELECT s.v AS v, '' AS sg, count() AS n, countIf(f.purchased::String != '') AS buys{feedback}
GROUP BY v""", 30)
        glob: dict[str, list[int]] = {}
        seg: dict[tuple[str, str], list[int]] = {}
        for r in rows:
            n, b = int(r["n"]), int(r["buys"])
            g = glob.setdefault(r["v"], [0, 0]); g[0] += n; g[1] += b
            if r["sg"]:
                x = seg.setdefault((r["v"], r["sg"]), [0, 0]); x[0] += n; x[1] += b
        self.glob = {k: (v[0], v[1]) for k, v in glob.items()}
        self.seg = {k: (v[0], v[1]) for k, v in seg.items()}
        self.report()

    def _posterior(self, v: str, sg: str | None) -> tuple[float, float]:
        # punto de partida: la tasa de la variante madre (las hijas heredan su reputación, con peso de 100 clientes)
        PN, PB = self.glob.get(PARENT, (0, 0))
        parent = (PB + 1) / (PN + 100)
        N, B = self.glob.get(v, (0, 0))
        a, b = parent * 100 + B, (1 - parent) * 100 + N - B   # tasa con todos los clientes
        mean = a / (a + b)
        if sg is None:
            return a, b
        n, b = self.seg.get((v, sg), (0, 0))
        return mean * self.PRIOR_STRENGTH + b, (1 - mean) * self.PRIOR_STRENGTH + n - b

    @staticmethod
    def candidates(sg: str | None) -> list[str]:
        """Variantes que compiten en ese tipo: las generales + la especializada de ese tipo (si existe)."""
        own = f"ep-seg-{SEG_VARIANT_FOR.get(sg, sg)}"
        return [v for v in VARIANTS if not v.startswith("ep-seg-") or v == own]

    def best(self, sg: str | None = None) -> str:
        def m(v):
            a, b = self._posterior(v, sg)
            return a / (a + b)
        return max(self.candidates(sg), key=m)

    def choose(self, sg: str | None = None) -> str:
        if FORCE_VARIANT:
            return FORCE_VARIANT
        if FINAL_MODE:
            return self.best(sg)
        return max(self.candidates(sg), key=lambda v: self.rng.betavariate(*self._posterior(v, sg)))

    def report(self) -> None:
        ctrl = self.glob.get("control", (0, 0))
        base = ctrl[1] / ctrl[0] if ctrl[0] else None
        stats = self.stats
        total = sum(n for n, _ in stats.values()) or 1
        print(f"\n[{time.strftime('%H:%M')}] Rendimiento en vivo (últimas {STATS_HOURS} h, compras reales):")
        print(f"  {'versión':<16}{'clientes':>9}{'compras':>9}{'% compra':>10}{'vs baseline':>13}{'% tráfico':>11}")
        if ctrl[0]:
            print(f"  {'control':<16}{ctrl[0]:>9}{ctrl[1]:>9}{base:>10.2%}{'—':>13}{'':>11}")
        for v, (n, b) in sorted(stats.items(), key=lambda x: -(x[1][1] / x[1][0] if x[1][0] else 0)):
            rate = b / n if n else 0.0
            lift = f"{(rate / base - 1):+.0%}" if base and n else "—"
            star = "  ← IA" if v == "v2-gbm" else ""
            print(f"  {v:<16}{n:>9}{b:>9}{rate:>10.2%}{lift:>13}{n / total:>11.0%}{star}")
        print("  Por tipo de cliente (variante elegida · sus compras/clientes en ese tipo · total del tipo):")
        for sg in SEGMENTS:
            n_sg = sum(n for (v, s_), (n, _) in self.seg.items() if s_ == sg and v in VARIANTS)
            b_sg = sum(b for (v, s_), (_, b) in self.seg.items() if s_ == sg and v in VARIANTS)
            bv = self.best(sg)
            nb, bb = self.seg.get((bv, sg), (0, 0))
            print(f"    {SEGMENT_NAMES[sg]:<22} → {bv:<16} {bb:>4}/{nb:<6}   tipo: {n_sg:>6} clientes, "
                  f"{(b_sg / n_sg if n_sg else 0):.2%} compra")
        print("  (pocas compras = mucha suerte: fiable a partir de ~50 compras por variante y tipo)\n")


BANDIT = Bandit()


# ---------------------------------------------------------------- bucle en vivo
NEW_REQUESTS = """
SELECT request_id::String AS request_id, user_id::String AS user_id, is_new_user,
       context.page::String AS page, context.device::String AS device, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60"""


_scratch_http = requests.Session()
_scratch_http.headers.update(http.headers)


def save_segments(segs: list[Record]) -> None:
    try:
        _scratch_http.post(f"{API}/v1/tables/scratch", params={"database": TEAM}, json=segs, timeout=10)
    except Exception as e:
        print(f"aviso: no se pudo guardar el tipo de cliente ({e})")


def refresher(model: Model, stop: threading.Event) -> None:
    last = defaultdict(float)
    every = {"catalog": 3, "trend": 60, "purchases": 60, "live_cooc": 300, "bandit": 120}
    jobs = {"bandit": BANDIT.refresh, "catalog": model.refresh_catalog, "trend": model.refresh_trend,
            "purchases": model.refresh_live_purchases, "live_cooc": model.refresh_live_cooc}
    while not stop.is_set():
        for name, fn in jobs.items():
            if time.time() - last[name] >= every[name]:
                last[name] = time.time()
                try:
                    fn()
                except Exception as e:
                    print(f"refresco {name} falló: {e}")
        time.sleep(0.5)


def main() -> None:
    model = Model()
    model.build_history()
    model.refresh_catalog()
    for fn in (model.refresh_trend, model.refresh_live_purchases, model.refresh_live_cooc):
        try:
            fn()
        except Exception as e:
            print(f"aviso: {fn.__name__} falló al arrancar: {e}")
    model.refresh_catalog()  # recalcula el orden con la tendencia ya cargada

    v2live = None
    if (Path(__file__).parent / "model_v2.pkl").exists():   # modelo entrenado con: python starter/v2.py train
        try:
            import v2
            v2live = v2.V2Live(model)
            VARIANTS["v2-gbm"] = None
            print("  variante v2-gbm añadida al bandit")
        except Exception:
            traceback.print_exc()
            print("aviso: no se pudo cargar la v2; seguimos con las variantes v1")
    try:
        BANDIT.refresh()
    except Exception as e:
        print(f"aviso: bandit sin datos todavía ({e})")
    print(f"Listo: {len(model.available)} productos disponibles. Respondiendo...")

    stop = threading.Event()
    threading.Thread(target=refresher, args=(model, stop), daemon=True).start()

    seen: set[str] = set()
    n_total, t_report = 0, time.time()
    while True:
        try:
            fresh = [r for r in query(NEW_REQUESTS) if r["request_id"] not in seen]
            answers, segs = [], []
            for req in fresh:
                seen.add(req["request_id"])
                try:
                    sg = segment(req, model)
                except Exception:
                    sg = "producto"
                if not FINAL_MODE and not FORCE_VARIANT and zlib.crc32(req["request_id"].encode()) % 1000 < SHARE_CONTROL * 1000:
                    arm = "control"
                else:
                    arm = BANDIT.choose(sg)
                try:
                    if arm == "control":
                        items = model.baseline()
                    elif arm == "v2-gbm":
                        items = v2live.recommend(req)
                    else:
                        items = model.recommend(req.get("recent_views"), req.get("user_id", ""), req.get("page", ""),
                                                VARIANTS[arm])
                    if len(set(items)) != 10:
                        raise ValueError(f"lista inválida: {items}")
                except Exception:
                    traceback.print_exc()
                    items = model.popular[:10]
                answers.append({"request_id": req["request_id"], "items": items, "model_version": arm})
                segs.append({"kind": "segment", "request_id": req["request_id"], "segment": sg})
            if answers:
                http.post(f"{API}/v1/tables/submissions", params={"database": TEAM},
                          json=answers, timeout=10).raise_for_status()
                n_total += len(answers)
                # tipo de cada cliente → scratch, en segundo plano (no retrasa las respuestas)
                threading.Thread(target=save_segments, args=(segs,), daemon=True).start()
            if time.time() - t_report > 30:
                print(f"{time.strftime('%H:%M:%S')} respondidas {n_total} peticiones")
                t_report = time.time()
        except Exception as e:  # nunca parar: los compradores siguen llegando
            print(f"error: {e}")
            time.sleep(1)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
