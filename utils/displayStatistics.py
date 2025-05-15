import pandas as pd
import plotly.express as px
import streamlit as st

def update_real_time_statistics(vehicle_stats_placeholder, accident_stats_placeholder, vehicle_speed_placeholder, plot_counter, vehicle_classes):
    vehicle_stats_df = pd.DataFrame(list(st.session_state.vehicle_count.items()), columns=["Class", "Count"])
    accident_stats_df = pd.DataFrame(list(st.session_state.accident_count.items()), columns=["Class", "Count"])

    # Filter DataFrame to remove rows with count 0
    vehicle_stats_df = vehicle_stats_df[vehicle_stats_df["Count"] > 0]
    accident_stats_df = accident_stats_df[accident_stats_df["Count"] > 0]

    # Show graph only if there is vehicle data
    if not vehicle_stats_df.empty:
        with vehicle_stats_placeholder.container():
            st.subheader("Vehicle Statistics")
            fig_vehicle = px.bar(vehicle_stats_df, x="Count", y="Class", orientation='h', title="Count of Vehicles per Class")
            st.plotly_chart(fig_vehicle, use_container_width=True, key=f"vehicle_stats_{plot_counter}")
            plot_counter += 1  # Increase counters for unique keys

    # Display graph only if there is accident data
    if not accident_stats_df.empty:
        with accident_stats_placeholder.container():
            st.subheader("Accident Statistics")
            fig_accident = px.bar(accident_stats_df, x="Count", y="Class", orientation='h', title="Count of Accidents per Class")
            st.plotly_chart(fig_accident, use_container_width=True, key=f"accident_stats_{plot_counter}")
            plot_counter += 1  # Increase counters for unique keys

    # Update real-time vehicle speed graph
    if st.session_state.vehicle_accident_data:
        df = pd.DataFrame(st.session_state.vehicle_accident_data)
        
        # Filter only for classes that belong to vehicle_classes
        df = df[df["Class"].isin(vehicle_classes)]

        if not df.empty:  # Do not create graph if there is no vehicle data
            # Calculate the average speed per class
            avg_speed_df = df.groupby("Class")["Speed (km/h)"].mean().reset_index()
            avg_speed_df.columns = ["Class", "Average Speed (km/h)"]

            # Graph the average speed
            fig_speed = px.bar(
                avg_speed_df, x="Average Speed (km/h)", y="Class",
                orientation='h', title="Average Speed of Vehicles per Class"
            )

            # Update graph in placeholder
            vehicle_speed_placeholder.empty()  # Clear placeholder
            vehicle_speed_placeholder.plotly_chart(fig_speed, key=f"vehicle_speed_{plot_counter}")
            plot_counter += 1  # Increase counters for unique keys

    return plot_counter