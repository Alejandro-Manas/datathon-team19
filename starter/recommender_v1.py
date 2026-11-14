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

import os
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

MODEL_VERSION = "v1-cooc"
Record = dict[str, Any]

# Test A/B en vivo: este % de peticiones recibe la v1; el resto, el baseline (grupo de control).
# Se reparte por hash del request_id, así que es aleatorio y reproducible.
# Comparar con: python starter/ab_report.py   ·   Para la ronda final (domingo): poner 1.0
SHARE_V1 = 0.5

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
}
DECAY = 0.6               # peso de la vista k-ésima más reciente: DECAY**k
ALREADY_BOUGHT = 0.15     # multiplicador para productos que el usuario ya compró
NEIGHBORS = 30


def query(sql: str, timeout: float = 10) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    r.raise_for_status()
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
SELECT s::String AS product_id, count() AS shown,
       countIf(ifNull(purchased::String, '') = s::String) AS bought
FROM shop.history_sessions ARRAY JOIN shown AS s
WHERE 1{c}
GROUP BY product_id"""


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
        self.subcat_top: dict[str, list[tuple[str, float]]] = {}
        self.available: set[str] = set()
        self.popular: list[str] = []      # orden de relleno, siempre válido

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
            q[r["product_id"]] = (float(r["bought"]) + 200 * 0.014) / (float(r["shown"]) + 200)
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

    def baseline(self) -> list[str]:
        return [p for p in self.best_sellers if p in self.available][:10]

    # ---- catálogo y stock (cada pocos segundos)
    def refresh_catalog(self, catalog_rows: list[Record] | None = None, out_of_stock: set[str] | None = None) -> None:
        rows = catalog_rows if catalog_rows is not None else query(SQL_CATALOG, 30)
        oos = out_of_stock if out_of_stock is not None else {r["product_id"] for r in query(SQL_OUT_OF_STOCK, 10)}
        available, subcat_of = set(), {}
        by_sub: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            p = r["product_id"]
            subcat_of[p] = r["subcategory"]
            if r["is_active"] in (True, 1, "true") and p not in oos:
                available.add(p)
                by_sub[r["subcategory"]].append(p)
        self.subcat_of = subcat_of
        self.subcat_top = {s: self._rank_norm(ps) for s, ps in by_sub.items()}
        self.popular = [p for p in sorted(available, key=lambda p: -self._prior(p))]
        self.available = available

    def _prior(self, p: str) -> float:
        return self.quality.get(p, 0.0) + 0.5 * self.trend.get(p, 0.0)

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

    def refresh_live_cooc(self) -> None:
        self.live_cooc = neighbors(query(SQL_LIVE_COOC, 120))

    def refresh_live_purchases(self) -> None:
        self.live_bought = {r["u"]: set(r["ps"]) for r in query(SQL_LIVE_PURCHASES, 60)}

    # ---- recomendación
    def recommend(self, recent_views: list[Record] | None, user_id: str = "", page: str = "") -> list[str]:
        avail = self.available
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

        for p in bought:
            if p in score:
                score[p] *= ALREADY_BOUGHT

        ranked = [p for p, _ in sorted(score.items(), key=lambda x: -x[1]) if p in avail]
        out = ranked[:10]
        if len(out) < 10:   # relleno: siempre 10 productos distintos y disponibles
            have = set(out)
            out += [p for p in self.popular if p not in have][: 10 - len(out)]
        return out


# ---------------------------------------------------------------- bucle en vivo
NEW_REQUESTS = """
SELECT request_id::String AS request_id, user_id::String AS user_id,
       context.page::String AS page, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60"""


def refresher(model: Model, stop: threading.Event) -> None:
    last = defaultdict(float)
    every = {"catalog": 3, "trend": 60, "purchases": 60, "live_cooc": 300}
    jobs = {"catalog": model.refresh_catalog, "trend": model.refresh_trend,
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
    print(f"Listo: {len(model.available)} productos disponibles. Respondiendo...")

    stop = threading.Event()
    threading.Thread(target=refresher, args=(model, stop), daemon=True).start()

    seen: set[str] = set()
    n_total, t_report = 0, time.time()
    while True:
        try:
            fresh = [r for r in query(NEW_REQUESTS) if r["request_id"] not in seen]
            answers = []
            for req in fresh:
                seen.add(req["request_id"])
                arm = MODEL_VERSION if zlib.crc32(req["request_id"].encode()) % 1000 < SHARE_V1 * 1000 else "control"
                try:
                    if arm == "control":
                        items = model.baseline()
                    else:
                        items = model.recommend(req.get("recent_views"), req.get("user_id", ""), req.get("page", ""))
                    if len(set(items)) != 10:
                        raise ValueError(f"lista inválida: {items}")
                except Exception:
                    traceback.print_exc()
                    items = model.popular[:10]
                answers.append({"request_id": req["request_id"], "items": items, "model_version": arm})
            if answers:
                http.post(f"{API}/v1/tables/submissions", params={"database": TEAM},
                          json=answers, timeout=10).raise_for_status()
                n_total += len(answers)
            if time.time() - t_report > 30:
                print(f"{time.strftime('%H:%M:%S')} respondidas {n_total} peticiones")
                t_report = time.time()
        except Exception as e:  # nunca parar: los compradores siguen llegando
            print(f"error: {e}")
            time.sleep(1)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
