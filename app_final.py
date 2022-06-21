import streamlit as st
#import pandas_profiling
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import signal


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=True)#False
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def create_plots():
    clist = df.columns.values

    channel = st.sidebar.selectbox("Select a channel:", clist)

    df['day'] = df.index
    fig = px.line(df, x='day', y=channel, title='Channel:' + channel)
    fig.update_xaxes(title_text='Observations')
    fig.update_yaxes(title_text='Channel: ' + channel)

    fig.update_layout(
        title_text="Blinking data"
    )

    # Add range slider

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)
    # fig4 = make_subplots(rows=1, cols=2)
    # fig4.add_trace(fig)

    # Creating the data for filteration
    T = 20000  # value taken in seconds
    n = 20000  # indicates total samples
    t = np.linspace(0, T, n, endpoint=False)

    # Filtering and plotting
    y = butter_lowpass_filter(df[channel], cutoff, fs, order)
    df['new'] = y
    fig1 = px.line(df, x='day', y='new', title='Channel: ' + channel)

    fig1.update_xaxes(title_text='Observations')
    fig1.update_yaxes(title_text='Channel: ' + channel)

    fig1.update_layout(
        title_text="Filtered Data"
    )

    # Add range slider

    fig1.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

    st.plotly_chart(fig1, use_container_width=True)
    return channel


def convert_df(df_convert):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df_convert.to_csv(index=False).encode('utf-8')

def decimate_data(df,channel):
    array = np.array(df[channel])
    ydem = signal.decimate(array, 15)

    decimate_df=pd.DataFrame(ydem)
    return (decimate_df)




if __name__ == "__main__":
    # order = 5
    # fs = 15
    # cutoff = 0.3
    # new params
    rad = 2 * np.pi
    order = 4
    cutoff = 15 * rad
    fs = 1000 * rad

    b, a = butter_lowpass(cutoff, fs, order)
    # Plotting the frequency response.
    w, h = freqz(b, a, worN=8000)

    st.set_page_config(layout="wide")
    uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file.seek(0)
        merge_df = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(merge_df,axis=0,ignore_index=True)
        st.sidebar.subheader('Save file as csv')
        merge = st.sidebar.button('Click here to save the file')
        if merge:
            merged_files = convert_df(df)
            st.sidebar.download_button(
                label="Download data as CSV",
                data=merged_files,
                file_name='merged_df.csv',
                mime='text/csv',
            )

        channel = create_plots()
        st.sidebar.subheader('Decimate Data')
        # st.title('Decimate Data')
        decimate = st.sidebar.button('Click here to Decimate the data')

        if decimate:
            df_convert = decimate_data(df, channel)
            csv = convert_df(df_convert)

            st.sidebar.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
            )




