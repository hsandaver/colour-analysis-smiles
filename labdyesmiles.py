import numpy as np

# Monkey patch for np.asscalar if it's not available
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

import streamlit as st
import pandas as pd
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# ------------------------------
# 1. App Configuration
# ------------------------------

st.set_page_config(
    page_title="🌈 Enhanced LAB Color Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# 2. Utility Functions
# ------------------------------

@st.cache_data
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load {uploaded_file.name}: {e}")
        return None

def validate_dataset(df, required_columns, filename):
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"CSV file '{filename}' is missing required columns: {', '.join(missing)}")
    # Convert 'L', 'A', 'B' to numeric if they exist
    for col in ['L', 'A', 'B']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Check for numeric validity in LAB columns
    if df[['L', 'A', 'B']].isnull().values.any():
        raise ValueError(f"Missing or invalid numeric values in 'L', 'A', or 'B' columns of '{filename}'.")
    return df

def calculate_delta_e(input_lab, dataset_df):
    input_color = LabColor(lab_l=input_lab[0], lab_a=input_lab[1], lab_b=input_lab[2])
    # Compute delta E for each row in the dataset
    delta_e = dataset_df.apply(
        lambda row: delta_e_cie2000(
            input_color,
            LabColor(lab_l=row['L'], lab_a=row['A'], lab_b=row['B'])
        ),
        axis=1
    )
    return delta_e

def lab_to_rgb(lab_color):
    try:
        lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
        rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
        # Clamp values between 0 and 1 and convert to 0-255 range
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
# 3. Main App Functionality
# ------------------------------

def main():
    st.title("🌈 **Enhanced LAB Color Analyzer** 🎨")
    st.markdown("""
    This application analyzes LAB color values and finds the closest matches from dye datasets.
    Upload your files, input LAB values, and explore the results with an interactive experience!
    """)

    # Sidebar File Upload
    st.sidebar.header("🗎 Upload Required Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV Files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload `iscc_nbs_lab_colors.csv` and `output_dye_colors_adjusted.csv`."
    )

    # Define required filenames and their expected columns
    required_files_info = {
        'iscc_nbs_lab_colors.csv': ['L', 'A', 'B', 'Color Name'],
        'output_dye_colors_adjusted.csv': ['L', 'A', 'B', 'Estimated Color', 'Chromophore', 'Corrected_SMILES']
    }

    datasets = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            # Only process files we expect
            if filename in required_files_info:
                df = load_csv(uploaded_file)
                if df is not None:
                    try:
                        required_cols = required_files_info[filename]
                        df = validate_dataset(df, required_cols, filename)
                        datasets[filename] = df
                        st.sidebar.success(f"✅ {filename} uploaded successfully.")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error with {filename}: {e}")
    else:
        st.sidebar.warning("⚠️ Please upload the required files to proceed.")
        st.stop()

    # Check if all required datasets are loaded
    missing_files = [fname for fname in required_files_info if fname not in datasets]
    if missing_files:
        st.warning(f"⚠️ Missing files: {', '.join(missing_files)}. Please upload them.")
        st.stop()

    # Retrieve validated datasets
    iscc_nbs_data = datasets['iscc_nbs_lab_colors.csv']
    dye_colors_df = datasets['output_dye_colors_adjusted.csv']

    # LAB Input Section
    st.header("🎨 Enter LAB Color Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        lab_l = st.number_input("L (Lightness):", min_value=0.0, max_value=100.0, value=50.0)
    with col2:
        lab_a = st.number_input("A (Green-Red):", min_value=-128.0, max_value=127.0, value=0.0)
    with col3:
        lab_b = st.number_input("B (Blue-Yellow):", min_value=-128.0, max_value=127.0, value=0.0)

    if st.button("🔍 Find Closest Matches"):
        input_lab = [lab_l, lab_a, lab_b]
        try:
            validate_lab_color(input_lab)
            # Calculate delta E values for the entire dye dataset
            delta_e_values = calculate_delta_e(input_lab, dye_colors_df)
            closest_idx = delta_e_values.idxmin()
            closest_dye = dye_colors_df.iloc[closest_idx]

            # Display Results
            st.subheader("🎨 Closest Dye Match")
            closest_rgb = lab_to_rgb([closest_dye['L'], closest_dye['A'], closest_dye['B']])
            st.markdown(f"**Estimated Color:** {closest_dye['Estimated Color']}")
            st.markdown(f"**Chromophore:** {closest_dye['Chromophore']}")
            st.markdown(f"**Delta-E:** {delta_e_values[closest_idx]:.2f}")
            st.markdown(f"**SMILES:** {closest_dye['Corrected_SMILES']}")

            # Display the color as a box
            color_box = f'''
                <div style="
                    width:100px;
                    height:50px;
                    background-color:rgb({closest_rgb[0]},{closest_rgb[1]},{closest_rgb[2]});
                    border:1px solid #000;">
                </div>
            '''
            st.markdown(color_box, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
