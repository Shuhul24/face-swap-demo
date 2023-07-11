#! /usr/bin/env python
import os
import cv2
import argparse
import streamlit as st
from face_detection import select_face, select_all_faces
from face_swap import face_swap
import gradio as gr

def main():
    st.set_page_config(page_title="Face Swap")
    st.header("Face Swapping")

    src_img = st.file_uploader("Source Image", type=["jpg", "jpeg", "png"])
    dst_img = st.file_uploader("Destination Image", type=["jpg", "jpeg", "png"])

    if src_img and dst_img:
    
        src_points, src_shape, src_face = select_face(src_img)

        dst_faceBoxes = select_all_faces(dst_img)

        if dst_faceBoxes is None:
            print('Detect 0 Face !!')
            exit(-1)

        output = dst_img
        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points,
                               dst_face["points"], dst_face["shape"],
                               output, "correct color")

        return output
if __name__ == '__main__':
    main()
    # demo = gr.Interface(fn=swapping, inputs=["image", "image"], outputs="image")
    # demo.launch()
