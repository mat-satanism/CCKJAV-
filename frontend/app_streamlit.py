# app_streamlit.py
import streamlit as st
import pandas as pd
import requests
import altair as alt
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"

st.set_page_config(page_title="Exoplanet Classifier (Demo)", layout="wide")
st.title("Exoplanet Classifier (Demo)")
st.write(
    "Введіть параметри транзиту та зорі або завантажте CSV з кількома об'єктами. "
    "Отримаєте прогноз класу (CONFIRMED / CANDIDATE / FALSE POSITIVE) та імовірності."
)
st.caption("Примітка: Це демо працює на тестовому сервісі. Точність відрізнятиметься від фінальної моделі.")

tab1, tab2 = st.tabs(["Інтерактивний прогноз", "Аналіз TESS (CSV)"])

FIELDS = [
    ("koi_period", "koi_period [days]", "Orbital period in days", 10.0, 0.2, 1000.0),
    ("koi_duration", "koi_duration [hours]", "Transit duration in hours", 3.0, 0.1, 30.0),
    ("koi_depth", "koi_depth [ppm]", "Transit depth in ppm", 800.0, 1.0, 200000.0),
    ("koi_prad", "koi_prad [R_earth]", "Planet radius in Earth radii", 2.5, 0.5, 30.0),
    ("koi_steff", "koi_steff [K]", "Stellar effective temperature", 5800.0, 2500.0, 10000.0),
    ("koi_slogg", "koi_slogg [log10(cm/s^2)]", "Stellar surface gravity", 4.4, 2.0, 5.5),
    ("koi_srad", "koi_srad [R_sun]", "Stellar radius in Solar radii", 1.0, 0.1, 50.0),
    ("koi_kepmag", "koi_kepmag [mag]", "Kepler/TESS magnitude", 15.5, 5.0, 18.0),
]

# -----------------------
# Tab 1 — single prediction
# -----------------------
with tab1:
    st.subheader("Інтерактивний прогноз (один об'єкт)")
    st.caption("Введіть 8 параметрів — натисніть 'Прогноз', щоб отримати ймовірності для трьох класів.")
    cols = st.columns(4)
    vals = {}
    for i, (key, label, hint, default, mn, mx) in enumerate(FIELDS):
        with cols[i % 4]:
            vals[key] = st.number_input(label, value=float(default), min_value=float(mn), max_value=float(mx), help=hint, format="%.4f")

    if st.button("Прогноз"):
        try:
            with st.spinner("Отримую прогноз..."):
                r = requests.post(f"{API}/predict_one", json=vals, timeout=10)
                if r.status_code != 200:
                    st.error(f"Помилка сервісу: {r.status_code} {r.text}")
                else:
                    data = r.json()
                    st.metric("Клас", data.get("pred_class", "-"))
                    probs = data.get("probs", {})
                    df = pd.DataFrame({"class": list(probs.keys()), "prob": [float(v) for v in probs.values()]})
                    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
                    df["prob_pct"] = (df["prob"] * 100).round(2)

                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("class:N", sort=None, title=None),
                        y=alt.Y("prob:Q", title="Probability"),
                        tooltip=[alt.Tooltip("prob_pct:Q", title="Probability (%)")]
                    )
                    st.altair_chart(chart.properties(height=300), use_container_width=True)

                    st.write("Точні значення (3 знаки):")
                    st.table(df.assign(prob=lambda d: d["prob"].round(3)).rename(columns={"prob": "probability"}))
        except requests.exceptions.RequestException as e:
            st.error("Сервіс тимчасово недоступний або таймаут. Перевірте URL у змінних оточення (API_URL).")

# -----------------------
# Tab 2 — batch + візуалізації
# -----------------------
with tab2:
    st.subheader("Аналіз TESS (батч)")
    st.caption("Завантажте CSV з мінімальними колонками: koi_period,koi_duration,koi_depth,koi_prad,koi_steff,koi_slogg,koi_srad,koi_kepmag")
    uploaded = st.file_uploader("CSV", type=["csv"])

    if uploaded is not None:
       try:
          file_bytes = uploaded.getvalue()
          df_in = pd.read_csv(io.BytesIO(file_bytes))
       except Exception as e:
          st.error(f"Не вдалося прочитати CSV: {e}")
          st.stop()

       expected = ["koi_period","koi_duration","koi_depth","koi_prad","koi_steff","koi_slogg","koi_srad","koi_kepmag"]
       missing = [c for c in expected if c not in df_in.columns]
       if missing:
          st.warning(f"Відсутні колонки: {', '.join(missing)}")
       else:
          try:
              files = {"file": (uploaded.name, io.BytesIO(file_bytes), "text/csv")}
              resp = requests.post(f"{API}/predict_batch", files=files, timeout=60)

              if resp.status_code != 200:
                  st.error(f"Помилка сервісу: {resp.status_code} {resp.text}")
              else:
                  preds_csv = resp.content.decode("utf-8")
                  out_df = pd.read_csv(io.StringIO(preds_csv))

                  st.write(f"Отримано {len(out_df)} рядків")
                  st.dataframe(out_df.head(50))

                  # --- Distribution bar chart (existing) ---
                  dist = out_df["pred_class"].value_counts().reset_index()
                  dist.columns = ["class", "count"]
                  chart = alt.Chart(dist).mark_bar().encode(
                      x="class", y="count", tooltip=["count"]
                  )
                  st.altair_chart(chart.properties(height=250), use_container_width=True)

                  st.download_button("Експорт в CSV", preds_csv, file_name="preds.csv")

                  # --- New visualizations block ---
                  st.markdown("---")
                  st.subheader("Візуалізації — P vs R, Correlation Heatmap, Boxplots")

                  # Convert numeric columns safely
                  numeric_cols = ["koi_period","koi_prad","koi_steff","koi_srad","koi_kepmag","koi_depth"]
                  for c in numeric_cols:
                      if c in out_df.columns:
                          out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

                  # 1) Scatter: Period vs Radius (log-log) colored by pred_class
                  if {"koi_period","koi_prad","pred_class"}.issubset(out_df.columns):
                      scatter_df = out_df.dropna(subset=["koi_period","koi_prad","pred_class"]).copy()
                      # to avoid issues with zeros or negatives, clip to small positive
                      scatter_df["koi_period_clipped"] = scatter_df["koi_period"].clip(lower=1e-3)
                      scatter_df["koi_prad_clipped"] = scatter_df["koi_prad"].clip(lower=1e-3)
                      fig_scatter = px.scatter(
                          scatter_df,
                          x="koi_period_clipped",
                          y="koi_prad_clipped",
                          color="pred_class",
                          hover_data=["kepid"] if "kepid" in scatter_df.columns else None,
                          labels={"koi_period_clipped": "Period (days)", "koi_prad_clipped": "Radius (R_earth)"},
                          title="Period vs Radius (кольором — pred_class)",
                          log_x=True,
                          log_y=True,
                          height=520
                      )
                      fig_scatter.update_layout(margin=dict(l=60, r=20, t=50, b=60))
                      st.plotly_chart(fig_scatter, use_container_width=True)
                  else:
                      st.info("Щоб побудувати P vs R потрібні колонки: koi_period, koi_prad, pred_class (є в відповіді API).")

                  # 2) Correlation heatmap between main numeric params
                  corr_cols = [c for c in numeric_cols if c in out_df.columns]
                  if len(corr_cols) >= 2:
                      corr_df = out_df[corr_cols].dropna()
                      # If dataset small, warn
                      if len(corr_df) < 5:
                          st.warning("Увага: дуже мало рядків для надійної кореляції (менше 5).")
                      corr_mat = corr_df.corr(method="pearson")
                      fig_heat = px.imshow(
                          corr_mat,
                          text_auto=True,
                          zmin=-1, zmax=1,
                          title="Кореляційна матриця (Pearson)",
                          labels=dict(x="Feature", y="Feature", color="Pearson r"),
                          height=480
                      )
                      fig_heat.update_layout(margin=dict(l=80, r=20, t=50, b=80))
                      st.plotly_chart(fig_heat, use_container_width=True)
                  else:
                      st.info("Не вистачає числових колонок для кореляції.")

                  # 3) Boxplots for key params by pred_class
                  box_features = ["koi_period","koi_prad","koi_steff"]
                  available_box = [f for f in box_features if f in out_df.columns]
                  if len(available_box) > 0 and "pred_class" in out_df.columns:
                      st.markdown("#### Boxplots (PLANET vs NOT-PLANET або інші класи)")
                      # Create grid of plots
                      cols_plot = st.columns( max(1, min(3, len(available_box))) )
                      for i, feat in enumerate(available_box):
                          with cols_plot[i % len(cols_plot)]:
                              bx_df = out_df.dropna(subset=[feat, "pred_class"])
                              if bx_df.empty:
                                  st.write(f"{feat}: недостатньо даних")
                                  continue
                              fig_box = px.box(
                                  bx_df,
                                  x="pred_class",
                                  y=feat,
                                  points="outliers",
                                  title=f"Boxplot — {feat} by pred_class",
                                  labels={feat: feat, "pred_class": "pred_class"},
                                  height=380
                              )
                              # if the feature is heavily skewed, offer log toggle
                              if (bx_df[feat] > 0).all():
                                  fig_box.update_yaxes(type="log")
                                  fig_box.update_layout(title=f"{feat} (log scale) by pred_class")
                              st.plotly_chart(fig_box, use_container_width=True)
                  else:
                      st.info("Для boxplots потрібні колонки: koi_period/koi_prad/koi_steff та pred_class.")

          except requests.exceptions.RequestException:
              st.error("Сервіс тимчасово недоступний або таймаут при відправці файлу.")

    with st.expander("Очікуваний шаблон CSV"):
        st.code("koi_period,koi_duration,koi_depth,koi_prad,koi_steff,koi_slogg,koi_srad,koi_kepmag\n2.0,3.1,850,1.9,5750,4.3,0.9,15.0")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Налаштування")
st.sidebar.text_input("API URL", API, key="api_url")
st.sidebar.markdown(
    "**Валідація полів:**\n"
    "- period: 0.2…1000 діб\n"
    "- duration: 0.1…30 год\n"
    "- depth: 1…200000 ppm\n"
    "- R_p: 0.5…30 R_earth\n"
    "- Teff: 2500…10000 K\n"
    "- logg: 2…5.5\n"
    "- R_star: 0.1…50 R_sun\n"
    "- Kepmag: 5…18"
)
