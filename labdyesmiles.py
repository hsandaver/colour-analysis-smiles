import numpy as np

# Temporary fix for older colormath versions referencing np.asscalar
if not hasattr(np, 'asscalar'):
    def asscalar(a):
        return a.item()
    np.asscalar = asscalar

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# ------------------------------
# 1. App Configuration
# ------------------------------

st.set_page_config(
    page_title="🌈 LAB Color Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# 2. Utility Functions
# ------------------------------

def validate_dataset(df, required_columns):
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")
    
    # Convert necessary columns to numeric
    for col in ['L', 'A', 'B']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'L' in df.columns and 'A' in df.columns and 'B' in df.columns:
        if df[['L', 'A', 'B']].isnull().values.any():
            raise ValueError("Missing or invalid numeric values in 'L', 'A', or 'B' columns.")
    return df

def calculate_delta_e(input_lab, dataset_lab):
    input_color = LabColor(lab_l=input_lab[0], lab_a=input_lab[1], lab_b=input_lab[2])
    delta_e = dataset_lab.apply(
        lambda row: delta_e_cie2000(
            input_color,
            LabColor(lab_l=row['L'], lab_a=row['A'], lab_b=row['B'])
        ),
        axis=1
    )
    return delta_e

def find_closest_color(input_lab, dataset_df):
    delta_e_values = calculate_delta_e(input_lab, dataset_df)
    min_idx = delta_e_values.idxmin()
    min_delta_e = delta_e_values[min_idx]
    closest_color = dataset_df.loc[min_idx]
    return closest_color, min_delta_e

def lab_to_rgb(lab_color):
    try:
        lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
        rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
        rgb_clamped = (
            int(max(0, min(rgb.rgb_r, 1)) * 255),
            int(max(0, min(rgb.rgb_g, 1)) * 255),
            int(max(0, min(rgb.rgb_b, 1)) * 255)
        )
        return rgb_clamped
    except Exception as e:
        st.error(f"❌ Error converting LAB to RGB: {e}")
        return (0, 0, 0)

def validate_lab_color(lab):
    if not isinstance(lab, (list, tuple, np.ndarray)) or len(lab) != 3:
        raise ValueError("Input LAB color must be a list, tuple, or array of three numerical values.")
    L, A, B = lab
    if not (0 <= L <= 100):
        raise ValueError("L component must be between 0 and 100.")
    if not (-128 <= A <= 127):
        raise ValueError("A component must be between -128 and 127.")
    if not (-128 <= B <= 127):
        raise ValueError("B component must be between -128 and 127.")

# ------------------------------
# 3. Dye Integration Functions with Top 3 Matches
# ------------------------------

# Expanded mapping of color descriptions to rough LAB values for dyes
dye_color_lab_map = {
    # Base Colors
    "red": [60, 70, 30],
    "orange": [75, 23, 78],
    "yellow": [97, -21, 94],
    "green": [46, -50, 50],
    "blue": [50, 5, -60],
    "violet": [35, 60, -60],
    "pink": [80, 30, 10],

    # Midpoints
    "orange-red": [67, 46, 54],  # Midpoint between Red and Orange
    "orange-yellow": [86, 1, 86],  # Midpoint between Orange and Yellow
    "cyan": [48, -25, -15],  # Midpoint between Green and Blue (Cyan/Teal)

    # Shifted Colors
    "shifted towards red (orange-yellow)": [86, 20, 75],
    "shifted towards red (orange)": [75, 35, 70],
    "shifted towards blue/violet (red-orange)": [67, 35, 25],
    "shifted towards blue/violet (orange-yellow)": [86, 10, 65],
    "shifted towards yellow (orange)": [75, 15, 85],
    "shifted towards red (cyan)": [48, 20, -10],
    "shifted towards blue/violet (cyan)": [48, 0, -55],
    "shifted towards yellow (cyan)": [48, -15, 50],
}

def find_top_dye_matches(input_lab, dye_colors_df, color_lab_map, top_n=3):
    dye_matches = []

    input_color = LabColor(lab_l=input_lab[0], lab_a=input_lab[1], lab_b=input_lab[2])

    for _, dye_row in dye_colors_df.iterrows():
        dye_color_desc = str(dye_row["Estimated Color"]).lower()
        matched = False
        for color, lab_val in color_lab_map.items():
            if color in dye_color_desc:
                dye_color_lab = LabColor(lab_l=lab_val[0], lab_a=lab_val[1], lab_b=lab_val[2])
                delta_e = delta_e_cie2000(input_color, dye_color_lab)
                dye_matches.append({
                    "Dye Color Description": dye_color_desc.capitalize(),
                    "Dye Name": dye_row.get("Chromophore", "Unknown Dye"),
                    "Dye SMILES": dye_row.get("Corrected_SMILES", "Unknown"),
                    "Dye LAB": lab_val,
                    "Delta-E": delta_e
                })
                matched = True
                break  # Stop after first match to avoid multiple entries
        if not matched:
            continue  # Skip dyes without matching color descriptions

    # Sort by Delta-E and return top N matches
    dye_matches_sorted = sorted(dye_matches, key=lambda x: x['Delta-E'])
    return dye_matches_sorted[:top_n]

def display_top_dye_matches(dye_matches):
    if not dye_matches:
        st.warning("No dye matches found.")
        return

    st.subheader(f"🎨 Top {len(dye_matches)} Dye Matches")
    for i, match in enumerate(dye_matches, start=1):
        col1, col2 = st.columns([1, 3])
        with col1:
            dye_rgb = lab_to_rgb(match['Dye LAB'])
            hex_color = '#%02x%02x%02x' % dye_rgb
            st.markdown(f"<div style='width:50px;height:50px;background-color:{hex_color}; border:1px solid #000;'></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{i}. {match['Dye Name']}**")
            st.markdown(f"*Description:* {match['Dye Color Description']}")
            st.markdown(f"*SMILES:* {match['Dye SMILES']}")
            st.markdown(f"*Delta-E:* {match['Delta-E']:.2f}")
            st.markdown("---")

# ------------------------------
# 4. Visualization Function
# ------------------------------

def plot_color_space(input_lab, closest_lab, dye_lab_list):
    fig = go.Figure()

    # Input Color
    fig.add_trace(go.Scatter3d(
        x=[input_lab[0]],
        y=[input_lab[1]],
        z=[input_lab[2]],
        mode='markers',
        marker=dict(size=8, color='black', symbol='circle'),
        name='Input Color'
    ))

    # Closest ISCC-NBS Color
    fig.add_trace(go.Scatter3d(
        x=[closest_lab[0]],
        y=[closest_lab[1]],
        z=[closest_lab[2]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='Closest ISCC-NBS Color'
    ))

    # Top Dye Matches
    for idx, dye_lab in enumerate(dye_lab_list, start=1):
        rgb = lab_to_rgb(dye_lab)
        hex_color = '#%02x%02x%02x' % rgb
        fig.add_trace(go.Scatter3d(
            x=[dye_lab[0]],
            y=[dye_lab[1]],
            z=[dye_lab[2]],
            mode='markers',
            marker=dict(size=6, color=hex_color, symbol='square'),
            name=f'Dye Match {idx}'
        ))

    # Set axes titles and ranges
    fig.update_layout(
        scene=dict(
            xaxis_title='L',
            yaxis_title='A',
            zaxis_title='B',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[-128, 127]),
            zaxis=dict(range=[-128, 127]),
        ),
        title='🌐 LAB Color Space Visualization',
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 5. Main App Functionality
# ------------------------------

def main():
    st.title("🌈 **Enhanced LAB Color Analyzer** 🎨")
    st.markdown("""
    This application allows you to analyze LAB color values, find the closest ISCC-NBS color, and discover the top 3 matching dyes based on color similarity. Upload your datasets, input LAB values, and explore the results interactively!
    """)

    # Sidebar for file uploads
    st.sidebar.header("📂 Upload Required Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV Files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload `iscc_nbs_lab_colors.csv` and `output_dye_colors_enhanced_with_corrections(1).csv`."
    )

    # Check if both required files are uploaded
    required_filenames = ['iscc_nbs_lab_colors.csv', 'output_dye_colors_enhanced_with_corrections(1).csv']
    datasets = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in required_filenames:
                try:
                    df = pd.read_csv(uploaded_file)
                    if uploaded_file.name == 'iscc_nbs_lab_colors.csv':
                        df = validate_dataset(df, ['L', 'A', 'B', 'Color Name'])
                    else:
                        df = validate_dataset(df, ['Estimated Color', 'Chromophore', 'Corrected_SMILES'])
                    datasets[uploaded_file.name] = df
                    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully.")
                except Exception as e:
                    st.sidebar.error(f"❌ Error with {uploaded_file.name}: {e}")

    if all(fname in datasets for fname in required_filenames):
        st.success("✅ All required files uploaded and validated successfully!")
    else:
        st.warning("⚠️ Please upload both `iscc_nbs_lab_colors.csv` and `output_dye_colors_enhanced_with_corrections(1).csv` to proceed.")
        st.stop()

    dataset_df = datasets['iscc_nbs_lab_colors.csv']
    dye_colors_df = datasets['output_dye_colors_enhanced_with_corrections(1).csv']

    # LAB Input Section
    st.header("🔍 Enter LAB Color Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        lab_l = st.number_input(
            label="L *Lightness*:",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.01,
            format="%.2f",
            help="Enter the Lightness value (0 to 100)."
        )
    with col2:
        lab_a = st.number_input(
            label="A *Green-Red*:",
            min_value=-128.0,
            max_value=127.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Enter the A value (-128 to 127). Positive values indicate red, negative indicate green."
        )
    with col3:
        lab_b = st.number_input(
            label="B *Blue-Yellow*:",
            min_value=-128.0,
            max_value=127.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Enter the B value (-128 to 127). Positive values indicate yellow, negative indicate blue."
        )

    # Action Button
    if st.button("🔍 Find Closest Color"):
        input_lab = [lab_l, lab_a, lab_b]
        try:
            validate_lab_color(input_lab)
            st.markdown(f"### 🟢 **Input LAB Color:** L={lab_l:.2f}, A={lab_a:.2f}, B={lab_b:.2f}")

            # Find closest ISCC-NBS color
            closest_color, delta_e = find_closest_color(input_lab, dataset_df)
            closest_color_name = closest_color['Color Name']
            closest_lab = [closest_color['L'], closest_color['A'], closest_color['B']]
            st.markdown(f"### 🎨 **Closest ISCC-NBS Color:** {closest_color_name}")
            st.markdown(f"**📏 Delta-E Value:** {delta_e:.2f}")

            # Convert to RGB
            input_rgb = lab_to_rgb(input_lab)
            closest_rgb = lab_to_rgb(closest_lab)
            input_hex = '#%02x%02x%02x' % input_rgb
            closest_hex = '#%02x%02x%02x' % closest_rgb

            # Display RGB Colors
            col_rgb1, col_rgb2 = st.columns(2)
            with col_rgb1:
                st.markdown("**🎨 Input RGB:**")
                st.markdown(f"<div style='width:100px;height:100px;background-color:{input_hex}; border:1px solid #000;'></div>", unsafe_allow_html=True)
                st.text(f"RGB: {input_rgb}\nHex: {input_hex}")
            with col_rgb2:
                st.markdown("**🎨 Closest RGB:**")
                st.markdown(f"<div style='width:100px;height:100px;background-color:{closest_hex}; border:1px solid #000;'></div>", unsafe_allow_html=True)
                st.text(f"RGB: {closest_rgb}\nHex: {closest_hex}")

            # Find top 3 closest dye matches
            top_dye_matches = find_top_dye_matches(input_lab, dye_colors_df, dye_color_lab_map, top_n=3)
            display_top_dye_matches(top_dye_matches)

            # Prepare dye LAB values for visualization
            dye_lab_list = [match['Dye LAB'] for match in top_dye_matches]
            plot_color_space(input_lab, closest_lab, dye_lab_list)

        except ValueError as ve:
            st.error(f"❌ Input validation error: {ve}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Developed with ❤️ using [Streamlit](https://streamlit.io/)**
    """)

if __name__ == "__main__":
    main()