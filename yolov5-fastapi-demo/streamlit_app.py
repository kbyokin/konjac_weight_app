import os
import sys
import cv2
import tempfile
from PIL import Image
import numpy as np

import streamlit as st
from streamlit.server.server import Server

# ? do dextr require config
config = ""
weight = ""

@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)

def trigger_rerun():
    session_infos = Server.get_current()._session_info_by_id.values()
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()

def load_dextr(config, weight):
    """ 
    wrapper func to load and cache dextr
    """
    dextr = "load from dextr.py"

    return dextr


def main():
    st.set_page_config(page_title="Konjac Weight Estimation")

    # state = SessionState.get(start = False, run = False)

    dextr = "" # !! load dextr --> construct dextr model return mask, width, height
    result = "" # !! 

    st.write("""
    # 1. Upload Konjac Images
    Folder or Single image
    """)
    upload = st.empty()
    start_button = st.empty()

    with upload:
        # receive an input
        file = st.file_uploader("Upload an image", accept_multiple_files=True)
        n_input = len(file)
        # print(n_input)

    # print(n_show)

    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(file.read())


        # display some input or assign
        n_show = st.slider('display only', min_value=0, max_value=n_input)
        n_cols = 5
        n_rows = 1 + n_input // 5
        rows = [st.beta_container() for _ in range(n_rows)]
        cols_per_row = [r.beta_columns(n_cols) for r in rows]

        for image_index, input_image in enumerate(file):
            if image_index != n_show:
                with rows[image_index // n_cols]:
                    cols_per_row[image_index // n_cols][image_index % n_cols].image(input_image)
                    st.button()
            else:
                break
        
        # if state.run:
        #     result = get_yolo([file(i) for i in range(n_input)])
        

st.write("""
# 2. Show results
--> Input image\
--> Mask image
--> Interval or Konjac result
""")

if __name__ == "__main__":
    main()