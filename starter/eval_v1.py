"""Evaluación offline de la v1 contra el baseline, sobre los últimos 7 días de shop.history_sessions.

    python starter/eval_v1.py              # tarda unos minutos (construye las señales con datos anteriores al corte)
    python starter/eval_v1.py --ablation   # además mide cuánto aporta cada señal (más lento)

Métricas (más alto = mejor):
  buy_hit@10   de las impresiones con compra, % en que nuestro top 10 contiene el producto comprado
  buy_ndcg     igual pero premiando ponerlo arriba (la posición importa)
  click_hit@10 lo mismo con los clics
  SNIPS_slot1  tasa de clic estimada del slot 1 corrigiendo el sesgo del sistema antiguo (ruidosa: mirar n_match)
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import recommender_v1 as v1  # noqa: E402

N_TEST = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 50000))


def load_test(cut: str) -> list[dict]:
    rows = v1.query(f"""
        SELECT user_id::String AS user_id, context.page::String AS page, recent_views, shown, propensity,
               ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased
        FROM shop.history_sessions WHERE ts >= '{cut}' ORDER BY rand() LIMIT {N_TEST}""", 600)
    return rows


def evaluate(policy, test) -> dict:
    buy_hit = buy_ndcg = click_hit = 0.0
    n_buy = n_click = 0
    w_sum = w_rew = 0.0
    n_match = 0
    slot_w, slot_r, slot_n = [0.0] * 10, [0.0] * 10, [0] * 10
    t = time.time()
    for r in test:
        recs = policy(r)
        if r["purchased"]:
            n_buy += 1
            if r["purchased"] in recs:
                buy_hit += 1
                buy_ndcg += 1 / math.log2(recs.index(r["purchased"]) + 2)
        if r["clicked"]:
            n_click += 1
            click_hit += r["clicked"] in recs
        for j in range(min(10, len(recs), len(r["shown"]))):   # IPS por hueco, con compras
            if recs[j] == r["shown"][j]:
                w = 1 / max(float(r["propensity"][j]), 1e-3)
                slot_w[j] += w
                slot_r[j] += w * (r["purchased"] == r["shown"][j])
                slot_n[j] += 1
        if recs and r["shown"] and recs[0] == r["shown"][0]:
            n_match += 1
            w = 1 / max(float(r["propensity"][0]), 1e-4)
            w_sum += w
            w_rew += w * (r["clicked"] == r["shown"][0])
    return {"buy_hit@10": round(buy_hit / max(n_buy, 1), 4), "buy_ndcg": round(buy_ndcg / max(n_buy, 1), 4),
            "click_hit@10": round(click_hit / max(n_click, 1), 4),
            "SNIPS_slot1": round(w_rew / w_sum, 4) if w_sum else None, "n_match": n_match,
            # compras esperadas por petición sumando los 10 huecos (aprox.: cada hueco por separado)
            "SNIPS_buy": round(sum(slot_r[j] / slot_w[j] for j in range(10) if slot_w[j]), 4),
            "buy_match": sum(slot_n),
            "ms/req": round(1000 * (time.time() - t) / max(1, len(test)), 2)}


def main() -> None:
    last_day = v1.query("SELECT toString(toDate(max(ts::DateTime64(3)))) AS d FROM shop.history_sessions", 120)[0]["d"]
    import datetime as dt
    cut = str(dt.date.fromisoformat(last_day) - dt.timedelta(days=7))
    print(f"entrenamiento < {cut} <= test")

    test = load_test(cut)
    print(f"{len(test)} impresiones de test, con compra: {sum(1 for r in test if r['purchased'])}")

    model = v1.Model()
    model.build_history(until=cut)
    catalog = v1.query("SELECT product_id::String AS product_id, subcategory::String AS subcategory, "
                       "true AS is_active FROM shop.catalog", 60)
    model.refresh_catalog(catalog_rows=catalog, out_of_stock=set())   # offline: todo el catálogo disponible

    results = {
        "baseline": evaluate(lambda r: model.baseline(), test),
        "v1": evaluate(lambda r: model.recommend(r["recent_views"], r["user_id"], r["page"]), test),
    }
    if "--sweep" in sys.argv:   # combinaciones de pesos: ¿más peso a lo popular/calidad?
        base = dict(v1.W)
        configs = {
            "calidad x3": {"quality": 2.1},
            "calidad x6": {"quality": 4.2},
            "calidad x12": {"quality": 8.4},
            "calidad x6 sin coview": {"quality": 4.2, "coview": 0.0},
            "calidad x6 sin coview/subcat": {"quality": 4.2, "coview": 0.0, "subcat": 0.0},
            "calidad x6 copurch x2": {"quality": 4.2, "coview": 0.0, "copurchase": 2.4},
            "calidad x6 self x3": {"quality": 4.2, "coview": 0.0, "self": 2.4},
        }
        for name, change in configs.items():
            v1.W.update(change)
            results[name] = evaluate(lambda r: model.recommend(r["recent_views"], r["user_id"], r["page"]), test)
            v1.W.clear()
            v1.W.update(base)
    if "--search" in sys.argv:   # búsqueda aleatoria de pesos; ordenada por SNIPS_buy
        import random
        rng = random.Random(19)
        base = dict(v1.W)
        grid = {"buy_after_view": [0, 1, 3, 6], "carousel": [0, 1, 2, 4], "coview": [0, 0.3, 1],
                "self": [0, 0.8, 2, 4], "subcat": [0, 0.3, 0.6, 1.5], "copurchase": [1.2, 2.4, 5, 10],
                "quality": [0, 0.3, 0.7, 1.5], "pop": [0, 1, 2, 4, 8]}
        found = []
        for i in range(int(next((a.split("=")[1] for a in sys.argv if a.startswith("--trials=")), 30))):
            cfg = {k: rng.choice(vals) for k, vals in grid.items()}
            v1.W.update(cfg)
            res = evaluate(lambda r: model.recommend(r["recent_views"], r["user_id"], r["page"]), test)
            found.append((res["SNIPS_buy"], cfg, res))
            print(f"  prueba {i + 1}: SNIPS_buy={res['SNIPS_buy']} buy_ndcg={res['buy_ndcg']}")
            v1.W.clear()
            v1.W.update(base)
        found.sort(key=lambda x: -x[0])
        for k, (_, cfg, res) in enumerate(found[:5]):
            results[f"top{k + 1}"] = res
            print(f"top{k + 1}: W = {cfg}")
    if "--ablation" in sys.argv:
        for name in list(v1.W):
            old = v1.W[name]
            v1.W[name] = 0.0
            results[f"v1 sin {name}"] = evaluate(lambda r: model.recommend(r["recent_views"], r["user_id"], r["page"]), test)
            v1.W[name] = old

    print()
    cols = ["buy_hit@10", "buy_ndcg", "click_hit@10", "SNIPS_slot1", "n_match", "SNIPS_buy", "buy_match", "ms/req"]
    print(f"{'política':<28}" + "".join(f"{c:>14}" for c in cols))
    for name, res in results.items():
        print(f"{name:<28}" + "".join(f"{str(res[c]):>14}" for c in cols))


if __name__ == "__main__":
    main()
