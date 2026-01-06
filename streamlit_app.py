import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# ============================================
# KONFIGURASI STREAMLIT
# ============================================
st.set_page_config(
    page_title="Call Centre Simulation",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📞 Simulasi Sistem Antrian Call Centre")
st.markdown("**Analisis What-If untuk Optimasi Jumlah Agent**",
            unsafe_allow_html=True)

# ============================================
# LOAD DATASET (CACHE untuk efisiensi)
# ============================================


@st.cache_data
def load_dataset():
    df_raw = pd.read_csv("dataset/simulated_call_centre.csv")
    service_times = (df_raw['service_length'] / 60).tolist()

    df_raw['call_started'] = pd.to_datetime(
        df_raw['call_started'], format='%I:%M:%S %p')
    df_raw = df_raw.sort_values('call_started')
    inter_arrivals = df_raw['call_started'].diff(
    ).dt.total_seconds().dropna() / 60
    inter_arrivals = inter_arrivals.tolist()

    return service_times, inter_arrivals, df_raw


service_times, inter_arrivals, df_raw = load_dataset()

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
st.sidebar.header("⚙️ Konfigurasi Simulasi")

# Kolom 1: Parameter Simulasi
col1, col2 = st.sidebar.columns(2)
with col1:
    sim_time = st.number_input(
        "Durasi Simulasi (menit)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Semakin lama durasi, semakin stabil hasil (tapi lebih lama proses)"
    )

with col2:
    random_seed = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=1000,
        value=42,
        help="Untuk reproducibility"
    )

st.sidebar.markdown("---")

# Parameter Skenario
st.sidebar.subheader("📊 Skenario yang Dibandingkan")

col1, col2 = st.sidebar.columns(2)
with col1:
    num_agents_a = st.number_input(
        "Skenario A: Jumlah Agents",
        min_value=1,
        max_value=10,
        value=1,
        help="Baseline scenario"
    )

with col2:
    num_agents_b = st.number_input(
        "Skenario B: Jumlah Agents",
        min_value=1,
        max_value=10,
        value=2,
        help="Improvement scenario"
    )

st.sidebar.markdown("---")

# Opsi Visualisasi
st.sidebar.subheader("📈 Pilih Visualisasi")
show_histogram = st.sidebar.checkbox("Histogram Waiting Time", value=True)
show_queue_evolution = st.sidebar.checkbox("Queue Evolution", value=True)
show_boxplot = st.sidebar.checkbox("Box Plot Comparison", value=True)

# ============================================
# FUNGSI SIMULASI
# ============================================


def run_simulation(num_agents, sim_time, seed):
    """
    Menjalankan simulasi call centre
    """
    random.seed(seed)
    records = []
    busy_time = 0
    queue_evolution = []

    def call_process(env, name, agent):
        nonlocal busy_time
        arrival = env.now

        with agent.request() as req:
            yield req
            start = env.now
            waiting_time = start - arrival

            service_duration = random.choice(service_times)
            busy_time += service_duration

            yield env.timeout(service_duration)
            finish = env.now

            records.append({
                "call_id": name,
                "arrival_time": arrival,
                "start_service": start,
                "finish_time": finish,
                "waiting_time": waiting_time,
                "service_time": service_duration,
                "system_time": finish - arrival
            })

    def call_generator(env, agent):
        i = 0
        while True:
            i += 1
            env.process(call_process(env, f"Call {i}", agent))
            queue_evolution.append({
                'time': env.now,
                'queue_length': len(agent.queue)
            })
            yield env.timeout(random.choice(inter_arrivals))

    # Jalankan simulasi
    env = simpy.Environment()
    agent = simpy.Resource(env, capacity=num_agents)
    env.process(call_generator(env, agent))
    env.run(until=sim_time)

    # Hasil
    df = pd.DataFrame(records)
    df_queue = pd.DataFrame(queue_evolution)
    utilization = (busy_time / (sim_time * num_agents)) * 100

    return df, df_queue, utilization


# ============================================
# MAIN CONTENT
# ============================================
st.markdown("## 1️⃣ Jalankan Simulasi")

if st.button("▶️ JALANKAN SIMULASI", use_container_width=True, type="primary"):
    with st.spinner("⏳ Menjalankan simulasi... (ini mungkin memakan waktu beberapa detik)"):
        try:
            # Simulasi Skenario A
            df_a, df_queue_a, util_a = run_simulation(
                num_agents_a, sim_time, random_seed)

            # Simulasi Skenario B
            df_b, df_queue_b, util_b = run_simulation(
                num_agents_b, sim_time, random_seed + 1)

            # Simpan ke session state
            st.session_state.df_a = df_a
            st.session_state.df_b = df_b
            st.session_state.df_queue_a = df_queue_a
            st.session_state.df_queue_b = df_queue_b
            st.session_state.util_a = util_a
            st.session_state.util_b = util_b
            st.session_state.simulation_done = True

            st.success("✅ Simulasi selesai!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# TAMPILKAN HASIL (jika simulasi sudah dijalankan)
# ============================================
if 'simulation_done' in st.session_state and st.session_state.simulation_done:

    # Ambil data dari session
    df_a = st.session_state.df_a
    df_b = st.session_state.df_b
    df_queue_a = st.session_state.df_queue_a
    df_queue_b = st.session_state.df_queue_b
    util_a = st.session_state.util_a
    util_b = st.session_state.util_b

    st.markdown("---")
    st.markdown("## 2️⃣ Hasil Simulasi")

    # ========== METRICS UTAMA ==========
    st.subheader("📊 Metrik Performa Utama")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"Scenario A: Calls Diproses",
            value=len(df_a),
            delta=f"+{len(df_b) - len(df_a)} calls (B)" if len(
                df_b) > len(df_a) else f"{len(df_b) - len(df_a)} calls (B)"
        )

    with col2:
        st.metric(
            label=f"Scenario A: Avg Wait Time",
            value=f"{df_a['waiting_time'].mean():.2f} min",
            delta=f"{((df_a['waiting_time'].mean() - df_b['waiting_time'].mean()) / df_a['waiting_time'].mean() * 100):.1f}%",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label=f"Scenario B: Calls Diproses",
            value=len(df_b)
        )

    with col4:
        st.metric(
            label=f"Scenario B: Avg Wait Time",
            value=f"{df_b['waiting_time'].mean():.2f} min"
        )

    st.markdown("---")

    # ========== TABEL PERBANDINGAN ==========
    st.subheader("📋 Tabel Perbandingan Detail")

    comparison_data = {
        'Metrik': [
            'Jumlah Calls Diproses',
            'Rata-rata Waiting Time (min)',
            'Maksimum Waiting Time (min)',
            'Median Waiting Time (min)',
            'Std Dev Waiting Time (min)',
            'Rata-rata System Time (min)',
            'Rata-rata Service Time (min)',
            'Utilization Agent (%)',
            'Min Waiting Time (min)',
            'Max Queue Length'
        ],
        f'Skenario A ({num_agents_a} Agent{"s" if num_agents_a > 1 else ""})': [
            len(df_a),
            f"{df_a['waiting_time'].mean():.2f}",
            f"{df_a['waiting_time'].max():.2f}",
            f"{df_a['waiting_time'].median():.2f}",
            f"{df_a['waiting_time'].std():.2f}",
            f"{df_a['system_time'].mean():.2f}",
            f"{df_a['service_time'].mean():.2f}",
            f"{util_a:.2f}",
            f"{df_a['waiting_time'].min():.2f}",
            f"{df_queue_a['queue_length'].max():.0f}"
        ],
        f'Skenario B ({num_agents_b} Agent{"s" if num_agents_b > 1 else ""})': [
            len(df_b),
            f"{df_b['waiting_time'].mean():.2f}",
            f"{df_b['waiting_time'].max():.2f}",
            f"{df_b['waiting_time'].median():.2f}",
            f"{df_b['waiting_time'].std():.2f}",
            f"{df_b['system_time'].mean():.2f}",
            f"{df_b['service_time'].mean():.2f}",
            f"{util_b:.2f}",
            f"{df_b['waiting_time'].min():.2f}",
            f"{df_queue_b['queue_length'].max():.0f}"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ========== VISUALISASI ==========
    st.subheader("📈 Visualisasi Data")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Histogram", "📈 Queue Evolution", "📦 Box Plot", "📉 Line Chart"])

    with tab1:
        if show_histogram:
            st.markdown(
                f"**Distribusi Waiting Time - Skenario A ({num_agents_a} Agent) vs B ({num_agents_b} Agent)**")

            col_a, col_b = st.columns(2)

            with col_a:
                fig_a = px.histogram(
                    df_a,
                    x='waiting_time',
                    nbins=30,
                    title=f'Skenario A: Histogram Waiting Time',
                    labels={
                        'waiting_time': 'Waiting Time (min)', 'count': 'Frequency'},
                    color_discrete_sequence=['#636EFA']
                )
                fig_a.add_vline(df_a['waiting_time'].mean(), line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {df_a['waiting_time'].mean():.2f}",
                                annotation_position="top right")
                st.plotly_chart(fig_a, use_container_width=True)

            with col_b:
                fig_b = px.histogram(
                    df_b,
                    x='waiting_time',
                    nbins=30,
                    title=f'Skenario B: Histogram Waiting Time',
                    labels={
                        'waiting_time': 'Waiting Time (min)', 'count': 'Frequency'},
                    color_discrete_sequence=['#00CC96']
                )
                fig_b.add_vline(df_b['waiting_time'].mean(), line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {df_b['waiting_time'].mean():.2f}",
                                annotation_position="top right")
                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Centang 'Histogram Waiting Time' di sidebar untuk menampilkan")

    with tab2:
        if show_queue_evolution:
            st.markdown(f"**Evolusi Jumlah Antrean Seiring Waktu**")

            fig_queue = go.Figure()

            fig_queue.add_trace(go.Scatter(
                x=df_queue_a['time'],
                y=df_queue_a['queue_length'],
                name=f'Skenario A ({num_agents_a} Agent)',
                mode='lines',
                fill='tozeroy',
                line=dict(color='#636EFA', width=2)
            ))

            fig_queue.add_trace(go.Scatter(
                x=df_queue_b['time'],
                y=df_queue_b['queue_length'],
                name=f'Skenario B ({num_agents_b} Agent)',
                mode='lines',
                fill='tozeroy',
                line=dict(color='#00CC96', width=2)
            ))

            fig_queue.update_layout(
                title='Queue Length Evolution',
                xaxis_title='Waktu Simulasi (menit)',
                yaxis_title='Jumlah Calls dalam Antrean',
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig_queue, use_container_width=True)
        else:
            st.info("Centang 'Queue Evolution' di sidebar untuk menampilkan")

    with tab3:
        if show_boxplot:
            st.markdown(f"**Perbandingan Distribusi Waiting & System Time**")

            # Prepare data untuk box plot
            box_data = pd.DataFrame({
                'Waiting Time (Scenario A)': df_a['waiting_time'],
                'Waiting Time (Scenario B)': df_b['waiting_time'],
                'System Time (Scenario A)': df_a['system_time'],
                'System Time (Scenario B)': df_b['system_time']
            })

            col_wait, col_sys = st.columns(2)

            with col_wait:
                fig_box_wait = go.Figure()
                fig_box_wait.add_trace(go.Box(
                    y=df_a['waiting_time'], name=f'Scenario A ({num_agents_a})', marker_color='#636EFA'))
                fig_box_wait.add_trace(go.Box(
                    y=df_b['waiting_time'], name=f'Scenario B ({num_agents_b})', marker_color='#00CC96'))
                fig_box_wait.update_layout(
                    title='Waiting Time Distribution', height=400)
                st.plotly_chart(fig_box_wait, use_container_width=True)

            with col_sys:
                fig_box_sys = go.Figure()
                fig_box_sys.add_trace(go.Box(
                    y=df_a['system_time'], name=f'Scenario A ({num_agents_a})', marker_color='#636EFA'))
                fig_box_sys.add_trace(go.Box(
                    y=df_b['system_time'], name=f'Scenario B ({num_agents_b})', marker_color='#00CC96'))
                fig_box_sys.update_layout(
                    title='System Time Distribution', height=400)
                st.plotly_chart(fig_box_sys, use_container_width=True)
        else:
            st.info("Centang 'Box Plot Comparison' di sidebar untuk menampilkan")

    with tab4:
        st.markdown(f"**Tren Waiting Time per Call**")

        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=df_a.index,
            y=df_a['waiting_time'],
            name=f'Skenario A ({num_agents_a} Agent)',
            mode='lines',
            line=dict(color='#636EFA', width=1.5),
            opacity=0.7
        ))

        fig_line.add_trace(go.Scatter(
            x=df_b.index,
            y=df_b['waiting_time'],
            name=f'Skenario B ({num_agents_b} Agent)',
            mode='lines',
            line=dict(color='#00CC96', width=1.5),
            opacity=0.7
        ))

        fig_line.update_layout(
            title='Waiting Time Trend per Call',
            xaxis_title='Call Order',
            yaxis_title='Waiting Time (min)',
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ========== ANALISIS & REKOMENDASI ==========
    st.markdown("## 3️⃣ Analisis & Rekomendasi")

    # Hitung improvement
    wait_improvement = ((df_a['waiting_time'].mean(
    ) - df_b['waiting_time'].mean()) / df_a['waiting_time'].mean() * 100)
    throughput_increase = ((len(df_b) - len(df_a)) / len(df_a) * 100)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Peningkatan Throughput (A→B)",
            f"{throughput_increase:.1f}%",
            f"+{len(df_b) - len(df_a)} calls"
        )

    with col2:
        st.metric(
            "Perubahan Avg Wait Time (A→B)",
            f"{wait_improvement:.1f}%",
            "decrease" if wait_improvement > 0 else "increase",
            delta_color="inverse" if wait_improvement > 0 else "off"
        )

    with col3:
        st.metric(
            "Utilization (A vs B)",
            f"{util_a:.1f}% vs {util_b:.1f}%",
            f"{abs(util_b - util_a):.1f}%"
        )

    # Analisis tekstual
    st.markdown("### 💡 Insight Utama:")

    insights = []

    # Insight 1: Throughput
    if throughput_increase > 0:
        insights.append(
            f"✅ **Throughput meningkat** {throughput_increase:.1f}% dengan penambahan agent (dari {num_agents_a} → {num_agents_b})")
    else:
        insights.append(
            f"⚠️ **Throughput menurun** {abs(throughput_increase):.1f}% - mungkin arrival rate tidak cukup tinggi")

    # Insight 2: Waiting Time
    if wait_improvement > 0:
        insights.append(
            f"✅ **Waiting time berkurang** {wait_improvement:.1f}% - sistem lebih efisien dengan agent tambahan")
    elif wait_improvement < -5:
        insights.append(
            f"⚠️ **Waiting time meningkat** {abs(wait_improvement):.1f}% - mungkin disebabkan lebih banyak calls diproses selama simulasi (paradoks statistik)")
    else:
        insights.append(
            f"➖ **Waiting time hampir sama** ({wait_improvement:.1f}%) - penambahan agent minimal berdampak")

    # Insight 3: Utilization
    if util_a > 100 or util_b > 100:
        insights.append(
            f"⚠️ **Sistem OVERLOAD** (utilization >100%) - demand terlalu tinggi untuk kapasitas yang ada")
        insights.append(
            f"💡 **Rekomendasi:** Tambah lebih banyak agents atau implementasikan teknologi alternatif (IVR, chatbot)")
    elif util_a > 80 or util_b > 80:
        insights.append(
            f"⚠️ **Sistem cukup sibuk** (utilization {max(util_a, util_b):.1f}%) - ada margin kecil untuk traffic spike")
    else:
        insights.append(
            f"✅ **Sistem seimbang** (utilization {max(util_a, util_b):.1f}%) - ada kapasitas untuk growth")

    for insight in insights:
        st.markdown(f"- {insight}")

    # Rekomendasi umum
    st.markdown("### 🎯 Rekomendasi untuk Manajer:")

    if throughput_increase > 0 and wait_improvement > 0 and util_b < 100:
        recommendation = f"""
        **Status: BAIK**
        - Penambahan dari {num_agents_a} → {num_agents_b} agent memberikan improvement signifikan
        - Waiting time berkurang {wait_improvement:.1f}% dan throughput meningkat {throughput_increase:.1f}%
        - **Rekomendasi:** Lakukan implementasi penambahan agent
        - **ROI Positif:** Cost agent tambahan < Benefit dari peningkatan kepuasan pelanggan
        """
    elif util_a > 100 or util_b > 100:
        recommendation = f"""
        **Status: KRITIS - SISTEM OVERLOAD**
        - Baik Scenario A maupun B masih overload (utilization >100%)
        - Penambahan dari {num_agents_a} → {num_agents_b} agent saja **TIDAK CUKUP**
        - **Rekomendasi URGENT:**
          1. Uji skenario dengan lebih banyak agents (3, 4, 5+)
          2. Implementasikan teknologi alternatif: IVR, chatbot, self-service
          3. Gunakan staffing fleksibel untuk peak hours
          4. Analisis apakah ada call yang bisa diredirect atau di-automate
        """
    else:
        recommendation = f"""
        **Status: MARGINAL**
        - Improvement dari penambahan agent dari {num_agents_a} → {num_agents_b} ada namun tidak signifikan
        - **Rekomendasi:**
          1. Uji dengan penambahan agent yang lebih besar (coba {num_agents_b + 2} atau lebih)
          2. Evaluasi alternative solutions (IVR, routing optimization, staffing schedule)
          3. Lakukan cost-benefit analysis dengan lebih detail
        """

    st.info(recommendation)

    st.markdown("---")

    # ========== DOWNLOAD LAPORAN ==========
    st.markdown("## 4️⃣ Download Laporan")

    # Buat CSV untuk download
    csv_comparison = comparison_df.to_csv(index=False).encode('utf-8')
    csv_scenario_a = df_a.to_csv(index=False).encode('utf-8')
    csv_scenario_b = df_b.to_csv(index=False).encode('utf-8')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download Perbandingan (CSV)",
            data=csv_comparison,
            file_name="comparison_results.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="📥 Download Scenario A (CSV)",
            data=csv_scenario_a,
            file_name=f"scenario_a_{num_agents_a}agents.csv",
            mime="text/csv"
        )

    with col3:
        st.download_button(
            label="📥 Download Scenario B (CSV)",
            data=csv_scenario_b,
            file_name=f"scenario_b_{num_agents_b}agents.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Silakan atur parameter di sidebar, lalu klik '▶️ JALANKAN SIMULASI' untuk memulai")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>
        📞 Call Centre Simulation Tool v1.0 | 
        Powered by SimPy & Streamlit | 
        TP-14 / TA-14 - Pemodelan dan Simulasi
    </small>
</div>
""", unsafe_allow_html=True)
