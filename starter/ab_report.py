"""Resultados del test A/B en vivo, a partir de team_19.feedback.

    python starter/ab_report.py          # últimas 2 horas
    python starter/ab_report.py 6        # últimas 6 horas
    python starter/ab_report.py 6 grupos # desglose por tipo de cliente

Cruza el feedback (qué pasó) con las submissions (qué versión respondió) por request_id.
La columna ±95% es el margen de error: si los intervalos de dos versiones no se solapan,
la diferencia es real y no suerte.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recommender_v1 import TEAM, query  # noqa: E402

hours = float(sys.argv[1]) if len(sys.argv) > 1 else 2
by_group = "grupos" in sys.argv

rows = query(f"""
SELECT {"concat(ifNull(c.sg, '?'), ' · ', s.model_version)" if by_group else "s.model_version"} AS arm, count() AS n,
       countIf(f.served_by::String = 'team') AS on_time,
       countIf(f.clicked::String != '') AS clicks,
       countIf(f.purchased::String != '') AS buys,
       sum(f.revenue_eur::Float64) AS revenue,
       quantile(0.95)(f.latency_ms::Float64) AS p95_ms
FROM (SELECT request_id::String AS request_id, served_by, clicked, purchased, revenue_eur, latency_ms
      FROM {TEAM}.feedback
      WHERE request_id::String != '__init__' AND ts::DateTime64(3) > now() - INTERVAL {int(hours * 60)} MINUTE) AS f
INNER JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS model_version
            FROM {TEAM}.submissions GROUP BY request_id) AS s
  ON f.request_id = s.request_id
LEFT JOIN (SELECT request_id::String AS request_id, any(segment::String) AS sg
           FROM {TEAM}.scratch WHERE kind::String = 'segment' GROUP BY request_id) AS c
  ON f.request_id = c.request_id
GROUP BY arm ORDER BY arm""", 60)

print(f"Últimas {hours:g} h\n")
print(f"{'versión':<32}{'peticiones':>11}{'a tiempo':>10}{'CTR':>8}{'compras %':>11}{'±95%':>8}{'€/petición':>12}{'p95 ms':>9}")
for r in rows:
    n = int(r["n"]) or 1
    p = int(r["buys"]) / n
    err = 1.96 * math.sqrt(p * (1 - p) / n)
    print(f"{r['arm']:<32}{n:>11}{int(r['on_time']) / n:>10.1%}{int(r['clicks']) / n:>8.2%}"
          f"{p:>11.2%}{err:>8.2%}{float(r['revenue']) / n:>12.3f}{float(r['p95_ms']):>9.0f}")
