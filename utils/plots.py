import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def PlotSignal(input_signal_df: pd.DataFrame, signal_field: str, lines = ['close']):
    fig = go.Figure()

    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,  # Adjust space between the plots
        row_heights=[1, 0.1]  # Adjust heights of the subplots
    )

    # Add price data
    for line in lines:
        fig.add_trace(go.Scatter(x=input_signal_df.index, 
                                y=input_signal_df[line], 
                                name=line),
                                row=1,
                                col=1)    


    if signal_field != "":
        buy_signals = input_signal_df[input_signal_df[signal_field] == 1]
        sell_signals = input_signal_df[input_signal_df[signal_field] == -1]
        # Add buy signals
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals[lines[0]],
            mode='markers',
            name='buy Signal',
            marker_symbol='triangle-up',
            marker_color='green',
            marker_size=10
        ),
            row=1,
            col=1)

        # Add sell signals
        fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals[lines[0]],
                mode='markers',
                name='sell Signal',
                marker_symbol='triangle-down',
                marker_color='red',
                marker_size=10
            ),  
                row=1, 
                col=1
            )
    
    # fig.add_trace(
    # go.Bar(x=input_signal_df.index, 
    #        y=input_signal_df['volume'],
    #        name='Volume'),
    #     row=2, col=1
    # )
    

    # Update layout
    fig.update_layout(
        title='buy and sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1000
    )

    fig.show()