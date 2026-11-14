"""Exploración rápida de los datos. Ejecutar desde la raíz del repo:

    python starter/explore.py > exploracion.txt

y pegar el contenido de exploracion.txt.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recommender import query  # noqa: E402


def show(title: str, sql: str, max_chars: int = 2500) -> None:
    print(f"\n===== {title}")
    try:
        rows = query(sql)
        print(json.dumps(rows, indent=1, default=str, ensure_ascii=False)[:max_chars])
    except Exception as e:
        print(f"ERROR: {e}")


# Ejemplos de cada tabla
show("shop.requests (2 últimas)", "SELECT * FROM shop.requests ORDER BY ts DESC LIMIT 2")
show("shop.catalog (2)", "SELECT * FROM shop.catalog LIMIT 2")
show("shop.history_sessions (2)", "SELECT * FROM shop.history_sessions LIMIT 2", 4000)
show("shop.history_events (3)", "SELECT * FROM shop.history_events LIMIT 3")
show("shop.events (3 últimos)", "SELECT * FROM shop.events ORDER BY ts DESC LIMIT 3")
show("shop.catalog_updates (3 últimos)", "SELECT * FROM shop.catalog_updates ORDER BY ts DESC LIMIT 3")
show("team_19.feedback (3 últimos)", "SELECT * FROM team_19.feedback ORDER BY ts DESC LIMIT 3")

# Tamaños
show("conteos", """
SELECT (SELECT count() FROM shop.catalog) AS catalog,
       (SELECT count() FROM shop.history_sessions) AS sessions,
       (SELECT count() FROM shop.history_events) AS hist_events,
       (SELECT count() FROM shop.requests) AS requests,
       (SELECT count() FROM shop.events) AS live_events,
       (SELECT count() FROM shop.catalog_updates) AS updates""")

# Estructura del catálogo y del comportamiento
show("productos por categoría", """
SELECT category::String AS category, count() AS n, round(avg(price_eur::Float64), 1) AS avg_price
FROM shop.catalog GROUP BY category ORDER BY n DESC""")
show("tipos de evento histórico", """
SELECT event_type::String AS t, count() AS n FROM shop.history_events GROUP BY t""")
show("tasas del carrusel antiguo", """
SELECT count() AS n,
       round(countIf(clicked::String != '') / count(), 4) AS ctr,
       round(countIf(purchased::String != '') / count(), 4) AS cvr,
       round(avg(revenue_eur::Float64), 3) AS rev_per_req
FROM shop.history_sessions""")
show("contexto de las peticiones en vivo", """
SELECT context.page::String AS page, is_new_user::Bool AS new_user,
       length(recent_views) AS n_views, count() AS n
FROM shop.requests GROUP BY page, new_user, n_views ORDER BY n DESC LIMIT 25""")
show("cambios de catálogo por tipo", """
SELECT change::String AS c, count() AS n FROM shop.catalog_updates GROUP BY c""")
show("nuestro feedback", """
SELECT served_by::String AS s, fallback_reason::String AS r, count() AS n,
       round(countIf(purchased::String != '') / count(), 4) AS cvr,
       round(avg(latency_ms::Float64)) AS lat
FROM team_19.feedback WHERE request_id::String != '__init__' GROUP BY s, r""")
