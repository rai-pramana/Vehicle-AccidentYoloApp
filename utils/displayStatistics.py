import pandas as pd
import plotly.express as px
import streamlit as st

def update_real_time_statistics(vehicle_stats_placeholder, accident_stats_placeholder, vehicle_speed_placeholder, plot_counter, vehicle_classes):
    vehicle_stats_df = pd.DataFrame(list(st.session_state.vehicle_count.items()), columns=["Kelas", "Jumlah"])
    accident_stats_df = pd.DataFrame(list(st.session_state.accident_count.items()), columns=["Kelas", "Jumlah"])

    # Filter DataFrame untuk menghapus baris dengan jumlah 0
    vehicle_stats_df = vehicle_stats_df[vehicle_stats_df["Jumlah"] > 0]
    accident_stats_df = accident_stats_df[accident_stats_df["Jumlah"] > 0]

    # Tampilkan grafik hanya jika ada data kendaraan
    if not vehicle_stats_df.empty:
        with vehicle_stats_placeholder.container():
            st.subheader("Statistik Kendaraan")
            fig_vehicle = px.bar(vehicle_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kendaraan per Kelas")
            st.plotly_chart(fig_vehicle, use_container_width=True, key=f"vehicle_stats_{plot_counter}")
            plot_counter += 1  # Tingkatkan counter untuk key yang unik

    # Tampilkan grafik hanya jika ada data kecelakaan
    if not accident_stats_df.empty:
        with accident_stats_placeholder.container():
            st.subheader("Statistik Kecelakaan")
            fig_accident = px.bar(accident_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kecelakaan per Kelas")
            st.plotly_chart(fig_accident, use_container_width=True, key=f"accident_stats_{plot_counter}")
            plot_counter += 1  # Tingkatkan counter untuk key yang unik

    # Update real-time vehicle speed graph
    if st.session_state.vehicle_accident_data:
        df = pd.DataFrame(st.session_state.vehicle_accident_data)
        
        # Filter hanya untuk kelas yang termasuk dalam vehicle_classes
        df = df[df["Kelas"].isin(vehicle_classes)]
        
        if not df.empty:  # Pastikan tidak membuat grafik jika tidak ada data kendaraan
            # Hitung rata-rata kecepatan per kelas
            avg_speed_df = df.groupby("Kelas")["Kecepatan (km/h)"].mean().reset_index()
            avg_speed_df.columns = ["Kelas", "Rata-Rata Kecepatan (km/h)"]
            
            # Buat grafik rata-rata kecepatan
            fig_speed = px.bar(
                avg_speed_df, x="Rata-Rata Kecepatan (km/h)", y="Kelas", 
                orientation='h', title="Rata-rata Kecepatan Kendaraan per Kelas"
            )
            
            # Perbarui grafik di placeholder
            vehicle_speed_placeholder.empty()  # Kosongkan placeholder
            vehicle_speed_placeholder.plotly_chart(fig_speed, key=f"vehicle_speed_{plot_counter}")
            plot_counter += 1  # Tingkatkan counter untuk key yang unik

    return plot_counter