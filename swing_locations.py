import streamlit as st
from streamlit import session_state as ss

import io
import itertools
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.lines as lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import time
import tqdm
import urllib

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from io import BytesIO
from matplotlib.patches import FancyBboxPatch, BoxStyle
from PIL import Image
from pyfonts import set_default_font, load_google_font
from scipy.stats import gaussian_kde
from typing import List
from zoneinfo import ZoneInfo

### Styling
base_font = 'DM Sans'
font = load_google_font(base_font, weight='bold')
regular_font = load_google_font(base_font, weight='regular')
italic = load_google_font(base_font, weight='bold', italic=True)
fm.fontManager.addfont(str(font.get_file()))

# Plot Style
pl_white = '#FFFFFF'
pl_background = '#262940'
pl_text = '#72CBFD'
pl_line_color = '#8D96B3'
pl_highlight = '#F1C647'
pl_highlight_gradient = ['#F1C647','#F5A05E']
pl_highlight_cmap = sns.color_palette(f'blend:{pl_highlight_gradient[0]},{pl_highlight_gradient[1]}', as_cmap=True)

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_line_color,
        'ytick.color': pl_line_color,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     },
    font=base_font
    )
mpl.rcParams.update({"font.weight": 400})
@st.cache_data(ttl=3600)
def letter_logo():
    logo_loc = 'https://res.cloudinary.com/dduabusaf/image/upload/v1772839606/teal_letter_logo_owufaj.png'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

letter_logo = letter_logo()

st.set_page_config(page_title='MLB Swing Location Charts', page_icon=letter_logo)
new_title = f'<p style="color:{pl_text}; font-weight: bold; font-size: 42px;">MLB Swing Location Charts</p>'
st.markdown(new_title, unsafe_allow_html=True)

group_colors = {
    'Fastball':'#f4707c',
    'Breaking_Ball':'#3da9f5',
    'Offspeed':'#34d399'
}

group_dict = {
    'FF':'Fastball',
    'SI':'Fastball',
    'FC':'Fastball',
    'CH':'Offspeed',
    'FS':'Offspeed',
    'FO':'Offspeed',
    'SL':'Breaking_Ball',
    'CU':'Breaking_Ball',
    'ST':'Breaking_Ball',
    'SV':'Breaking_Ball',
    'KN':'Breaking_Ball'
}

marker_colors = {
    'FF':'#FF6683',
    'SI':'#F2B24B',
    'FS':'#83D6FF',
    'FC':'#C59C9C',
    'SL':'#CE66FF',
    'ST':'#FFAAF7',
    'CU':'#339cff',
    'CS':'#2A98FF',
    'SV':'#2A98FF',
    'CH':'#6DE95D',
    'SC':'#6DE95D',
    'KN':'#c7c7c7',
    'SC':'#c7c7c7',
    'UN':'#c7c7c7',
}

marker_names = {
    'FF':'4-Seam',
    'SI':'Sinker',
    'FO':'Forkball',
    'FS':'Splitter',
    'FC':'Cutter',
    'SL':'Slider',
    'ST':'Sweeper',
    'CU':'Curveball',
    'CS':'Slow Curve',
    'SV':'Slurve',
    'CH':'Changeup',
    'SC':'Screwball',
    'KN':'Knuckleball',
    'UN':'Unknown',
}

def pull_day_misses(day):
    date_dfs = []
    day_text = day.strftime('%Y-%m-%d')
    for pos in ['batter','pitcher']:
        url_miss_dist = f"https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-timing-miss-distance?type={pos}&season%5B%5D=2026&splitYear=1&min=1&split%5B%5D=api_pitch_type_group03&split%5B%5D=pitch_hand&split%5B%5D=bat_side&split%5B%5D=pitchzone_height_code&minSplit=1&gameType%5B%5D=R&dateStart={day_text}&dateEnd={day_text}&batSide=&contactType=&attackZone=&pitchHand=&showColumn=z&csv=true"
        res_miss_dist = requests.get(url_miss_dist, timeout=None).content
        date_dfs += [pd.read_csv(io.StringIO(res_miss_dist.decode('utf-8'))).assign(game_date = day, pos=pos)]
    return pd.concat(date_dfs,ignore_index=True).reset_index()

@st.cache_data(ttl=3600,show_time=True)
def load_timing_data():
    date_dfs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(pull_day_misses, day): day for day in pd.date_range(start=datetime(2026, 3, 25), end=datetime.now(ZoneInfo("America/New_York")).date()-timedelta(days=1))}
        for future in as_completed(futures):
            date_dfs += [future.result()]
    raw_data = pd.concat(date_dfs,ignore_index=True).reset_index()
    miss_concise_df = raw_data[['game_date','id', 'name','pos','bat_side', 'pitch_hand',
                                    'api_pitch_type', 'pitchzone_height_code',
                                    'miss_distance', 'flawed_percent','perfect_percent',
                                    'tied_up_percent', 'avg_x_tied_up','centered_percent', 'flailed_percent', 'avg_x_flail',
                                    'early_percent', 'avg_y_early', 'on_time_percent', 'late_percent','avg_y_late',
                                    'over_percent', 'avg_z_over', 'lined_up_percent', 'under_percent','avg_z_under',
                                    'n_swings']].rename(columns={'avg_x_flail':'avg_x_flailed'}).copy()
    in_out = ['tied_up', 'centered', 'flailed']
    over_under = ['over', 'lined_up', 'under']
    early_late = ['early', 'on_time', 'late']
    timing_types = in_out + over_under + early_late

    miss_concise_df['avg_x_centered'] = None
    miss_concise_df['avg_y_on_time'] = None
    miss_concise_df['avg_z_lined_up'] = None
        
    miss_concise_df['perfect'] = miss_concise_df['perfect_percent'].mul(miss_concise_df['n_swings']).round(0).astype('int')
    for letter in ['x','y','z']:
        miss_concise_df[f'avg_{letter}_perfect'] = None
    for stat in timing_types:
        miss_concise_df[stat] = miss_concise_df[stat+'_percent'].mul(miss_concise_df['n_swings']).round(0).astype('int')
    return miss_concise_df
    
timing_data = load_timing_data()
hitter_list = [' '.join(x.split(', ')[::-1]) for x in timing_data.loc[timing_data['pos']=='batter','name'].sort_values(key=lambda x: x.str.lower()).unique()]
pitcher_list = [' '.join(x.split(', ')[::-1]) for x in timing_data.loc[timing_data['pos']=='pitcher','name'].sort_values(key=lambda x: x.str.lower()).unique()]
timing_data['name'] = [' '.join(x.split(', ')[::-1]) for x in timing_data['name']]

def player_bio_data(player_id: int) -> List[str]:
    """
    Fetches the names of players given their IDs.

    Parameters:
    player_ids: List of player IDs.

    Returns:
    List of player name, batSide, pitchHand.
    """
    url = f'https://statsapi.mlb.com/api/v1/people?personIds={player_id}'

    response = requests.get(url)

    # raise an HTTPError if the request was unsuccessful
    response.raise_for_status()

    json_data = response.json()['people'] # parse the JSON response
    return [json_data[0]['nameFirstLast'], json_data[0]['batSide']['code'], json_data[0]['pitchHand']['code']]

def combine_counts(x,y,pitchtype,p_hand,b_hand):
    if x.shape[0]==0:
        group_mul = 1
    else:
        group_mul = x['count'].sum()**2

    return (
        pd.DataFrame(itertools.product(x.loc[x.index.repeat(x['count'])]['distance'],
                                       y.loc[y.index.repeat(y['count'])]['distance']),
                     columns=['in_out','over_under'])
        .assign(pitchtype = pitchtype,
                p_hand = p_hand,
                b_hand = b_hand,
                mul = group_mul)
        )

def load_logo():
    logo_loc = 'https://res.cloudinary.com/dduabusaf/image/upload/v1772839288/PitcherList_Stats_watermark_with_logo_k9e3xa.webp'
    with urllib.request.urlopen(logo_loc) as response:
        logo = Image.open(response)

    return logo

logo = load_logo()

def player_miss_data(input_fields, player_io_df, player_ou_df):
    date, pitch_type, zone_height, stand, throw = input_fields
    test_io = player_io_df.loc[(player_io_df['game_date']==date) &
                                (player_io_df['api_pitch_type']==pitch_type) &
                                (player_io_df['pitchzone_height_code']==zone_height) &
                                (player_io_df['bat_side']==stand) &
                                (player_io_df['pitch_hand']==throw)
                                ].reset_index(drop=True)
    io_dict = test_io.set_index('variable')['count'].to_dict()
    if io_dict:
        if (io_dict['flailed'] + io_dict['tied_up'])==0:
            net_out_val = 0
        else:
            net_out_val = (io_dict['flailed'] - io_dict['tied_up']) / (io_dict['flailed'] + io_dict['tied_up'])
        test_io['distance'] = test_io['distance'].astype('float').fillna(np.clip(pd.Series(np.random.normal(4.0*net_out_val,4/3,size=test_io.shape[0])),-4,4).astype('float'))

    test_ou = player_ou_df.loc[(player_ou_df['game_date']==date) &
                                (player_ou_df['api_pitch_type']==pitch_type) &
                                (player_ou_df['pitchzone_height_code']==zone_height) &
                                (player_ou_df['bat_side']==stand) &
                                (player_ou_df['pitch_hand']==throw)
                                ].reset_index(drop=True)
    ou_dict = test_ou.set_index('variable')['count'].to_dict()
    if ou_dict:
        if (ou_dict['under'] + ou_dict['over'])==0:
            net_over_val = 0
        else:
            net_over_val = (ou_dict['under'] - ou_dict['over']) / (ou_dict['under'] + ou_dict['over'])
        test_ou['distance'] = test_ou['distance'].astype('float').fillna(np.clip(pd.Series(np.random.normal(2.0*net_over_val,2/3,size=test_ou.shape[0])),-2,2).astype('float'))

    return combine_counts(test_io,test_ou,pitch_type,throw,stand)

def expand_data(input_list, player_io_df, player_ou_df):
    io_ou_combos = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(player_miss_data, input_values, player_io_df, player_ou_df): input_values for input_values in input_list}
        for future in as_completed(futures):
            io_ou_combos += [future.result()]
    combo_df = pd.concat(io_ou_combos,ignore_index=True)
    combo_df['group'] = combo_df['pitchtype'].map(group_dict)
    combo_df['color'] = combo_df['group'].map(group_colors)

    return combo_df

def transform_data(player_id,base_df,pos_text,p_hand,b_hand):
    group_cols = ['id','name','pos','game_date','api_pitch_type','pitchzone_height_code','bat_side','pitch_hand']
    in_out = ['tied_up', 'centered', 'flailed']
    over_under = ['over', 'lined_up', 'under']
    
    player_df = base_df.loc[(base_df['id']==player_id) & (base_df['pos']==pos_text) & (base_df['pitch_hand']==p_hand) & (base_df['bat_side']==b_hand)]
    player_io_df = (
        pd.merge(
            player_df.melt(
                id_vars=group_cols,
                value_vars=in_out,
                value_name='count'
            ),
            player_df.melt(
                id_vars=group_cols,
                value_vars=['avg_x_'+x for x in in_out],
                value_name='distance'
            ).assign(variable =lambda x: x['variable'].str.replace('avg_x_','')),
            how='inner',
            on=group_cols+['variable'])
        .sort_values(['game_date','id','pos','api_pitch_type','pitchzone_height_code','bat_side','pitch_hand','variable'])
        .reset_index(drop=True)
    )

    player_ou_df = (
        pd.merge(
            player_df.melt(
                id_vars=group_cols,
                value_vars=over_under,
                value_name='count'),
            player_df.melt(
                id_vars=group_cols,
                value_vars=['avg_z_'+x for x in over_under],value_name='distance'
            ).assign(variable =lambda x: x['variable'].str.replace('avg_z_','')),
            how='inner',
            on=group_cols+['variable'])
        .sort_values(['game_date','id','pos','api_pitch_type','pitchzone_height_code','bat_side','pitch_hand','variable'])
        .reset_index(drop=True)
    )

    if pos=='b':
      swing_count_dict = player_df.groupby(['api_pitch_type','pitch_hand'])['n_swings'].sum().to_dict()
    else:
      swing_count_dict = player_df.groupby(['api_pitch_type','bat_side'])['n_swings'].sum().to_dict()
    input_list = list(player_io_df[['game_date','api_pitch_type','pitchzone_height_code','bat_side','pitch_hand']].value_counts().index)

    return swing_count_dict, expand_data(input_list, player_io_df, player_ou_df)

def miss_chart(combo_df,pos,b_hand,p_hand):
    group_base = 'pitchtype' if pos=='p' else 'group'
    chart_colors = marker_colors if pos=='p' else group_colors
    if pos=='b':
        b_hand = combo_df.loc[combo_df['p_hand']==p_hand,'b_hand'].value_counts().index[0]
    for group in list(combo_df.loc[(combo_df['p_hand']==p_hand) & (combo_df['b_hand']==b_hand),group_base].value_counts().index):
        if pos=='p':
            bw_adjust = np.clip((swing_count_dict[(group,b_hand)]-10)/20,0.5,1.75)
            if swing_count_dict[(group,b_hand)]<20:
                continue
        else:
            bw_adjust = 1.1
        chart_df = combo_df.loc[(combo_df[group_base]==group) & (combo_df['p_hand']==p_hand) & (combo_df['b_hand']==b_hand)].reset_index(drop=True)
        chart_df = chart_df.loc[chart_df.index.repeat(chart_df['mul'].max()**0.5-chart_df['mul']**0.5+1)].reset_index(drop=True)

        # gauss_data = np.vstack([chart_df['in_out'],chart_df['over_under']])
        # kde = gaussian_kde(gauss_data)
        # densities = kde(gauss_data)
        # chart_mode = gauss_data[:, densities.argmax()]

        # chart_mean = (chart_df['in_out'].mean(),chart_df['over_under'].mean())

        fig, ax = plt.subplots(figsize=(8,4))
        if pos=='p':
            fig.text(0.5,0.83,f"{b_hand}HH vs {player_name}'s {marker_names[group]}",ha='center',va='center')
        else:
            plural_text = '' if group =='Offspeed' else 's'
            fig.text(0.5,0.83,f"{player_name} vs {group.replace('_',' ')}{plural_text} ({p_hand}HP)",
                    ha='center',va='center')
        # fig.text(0.175 if b_hand=='R' else 0.675,0.175,
        #          f"x = Average\n+ = Most Common",
        #          ha='left',va='center',
        #          fontsize=11,font=regular_font,
        #          alpha=0.5)
        fig.add_artist(lines.Line2D([0.125, 0.125], [0.11, 0.95],linewidth=3,color=chart_colors[group]))
        fig.add_artist(lines.Line2D([0.895, 0.895], [0.11, 0.95],linewidth=3,color=chart_colors[group]))
        fig.add_artist(lines.Line2D([0.125, 0.895], [0.11, 0.11],linewidth=3,color=chart_colors[group]))
        fig.add_artist(lines.Line2D([0.125, 0.895], [0.95, 0.95],linewidth=3,color=chart_colors[group]))
        sns.kdeplot(chart_df,
                    x='in_out',
                    y='over_under',
                    cmap=sns.blend_palette([pl_background,chart_colors[group]],as_cmap=True),
                    bw_adjust=bw_adjust,
                    cut=2,
                    levels=10,
                    fill=True)

        ### Plot most common/average distance
        # ax.plot(chart_mode[0],chart_mode[1],marker='x',color='w',markersize=10,linewidth=2,alpha=0.75)
        # ax.plot(chart_mean[0],chart_mean[1],marker='x',color='w',markersize=10,linewidth=2,alpha=0.75)
        
        ### Bat
        ## Top Line
        ax.plot([-6,5.25], [2.61/2,2.61/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        ax.plot([-26.75,-12.125], [1/2,0.9/2], color='w', zorder=9, linewidth=2, alpha=0.5)


        ## Bottom Line
        ax.plot([-26.75,-12.125], [-1/2,-1.1/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        ax.plot([-6,5.25], [-2.61/2,-2.61/2], color='w', zorder=9, linewidth=2, alpha=0.5)

        if b_hand=='R':
            ax.plot([5.4,5.95], [2.59/2,2.36/2], color='w', zorder=9, linewidth=2, alpha=0.5)
            ax.plot([-12,-6.105], [0.9/2,2.58/2], color='w', zorder=9, linewidth=2, alpha=0.5)

            ax.plot([5.4,5.95], [-2.62/2,-2.36/2], color='w', zorder=9, linewidth=2, alpha=0.5)
            ax.plot([-12,-6.105], [-1.1/2,-2.64/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        else:
            ax.plot([5.37,5.95], [2.58/2,2.36/2], color='w', zorder=9, linewidth=2, alpha=0.5)
            ax.plot([-12,-6.125], [0.9/2,2.58/2], color='w', zorder=9, linewidth=2, alpha=0.5)

            ax.plot([5.37,5.95], [-2.63/2,-2.36/2], color='w', zorder=9, linewidth=2, alpha=0.5)
            ax.plot([-12,-6.125], [-1.1/2,-2.64/2], color='w', zorder=9, linewidth=2, alpha=0.5)
            
        # Cap
        ax.plot([6,6], [-2.15/2,2.25/2], color='w', zorder=9, linewidth=2, alpha=0.5)

        ## Handle
        ax.plot([-27,-27], [-2/2,2/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        ax.plot([-28,-28], [-2/2,2/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        ax.plot([-28,-27], [-2/2,-2/2], color='w', zorder=9, linewidth=2, alpha=0.5)
        ax.plot([-28,-27], [2/2,2/2], color='w', zorder=9, linewidth=2, alpha=0.5)

        ## Barrel Area
        fig.add_artist(mpatches.FancyBboxPatch((0.42, 0.455), 0.185, 0.08,
                                                ec='w',
                                                fc='w',
                                                alpha=0.1,
                                                zorder=8,
                                                linewidth=1,
                                                boxstyle=mpatches.BoxStyle("Round", pad=0.2),
                                                mutation_scale=0.05
                                                      )
                )

        ax.set(xlim=(-15,15) if b_hand=='R' else (15,-15),
                ylim=(-7.5,7.5),aspect=1)

        fig.suptitle('Ball Locations Relative to the Bat',y=0.94,fontsize=20)
        ax.axis('off')
        logo_ax = fig.add_axes([0.38,0.13,0.24,0.24], anchor='SW', zorder=1)
        logo_ax.imshow(logo)
        logo_ax.axis('off')
        sns.despine(trim=True)
        try:
            st.pyplot(fig, width='content')
        except PIL.Image.DecompressionBombError:
            st.write('Error generating chart. Please click "Generate Charts" button again')

if 'pos' not in ss:
    ss['pos'] = 'Pitcher'
    
def pos_change():
    if 'player' in ss:
        del ss['player']
col1, col2, col3 = st.columns([0.2,0.35,0.45])
with col1:
    st.radio("Select a position:", ['Pitcher','Batter'],index=0, 
             key='pos',on_change=pos_change)

pos_text = ss['pos'].lower()
pos = pos_text[0]



if 'player' not in ss:
    if ss['pos'] == 'Pitcher':
        ss['player'] = 'Cristopher Sánchez'
    else:
        ss['player'] = 'Juan Soto'

if ss['pos'] == 'Pitcher':
    player_list = pitcher_list
else:
    player_list = hitter_list
player_idx = player_list.index(ss['player'])

with col2:
    st.selectbox(f'Choose a {pos_text}:',
                 player_list,
                 index=player_idx,
                 key='player')
    player_ids = list(timing_data.loc[(timing_data['pos']==pos_text) & (timing_data['name']==ss['player']),'id'].value_counts().index)
    if len(player_ids)==1:
        player_id = player_ids[0]
    else:
        # For your Maxes Muncy, your Luises Castillo, etc
        player_id = st.selectbox(f'Choose an MLBAMID:',player_ids)

player_name, b_hand, p_hand = player_bio_data(player_id)

hitter_handednesses = ['R','L']
pitcher_handednesses = ['R','L']
with col3:
    if pos=='p':
        if p_hand=='S':
            p_hand = st.selectbox(f"Choose the player's pitching side:",['R','L'])
        hitter_handednesses = list(timing_data.loc[(timing_data['pos']==pos_text) & (timing_data['id']==player_id) & (timing_data['pitch_hand']==p_hand),'bat_side'].value_counts().index)
        b_hand = st.selectbox(f"Choose the opposing hitters' handedness:",hitter_handednesses)
    else:
        if b_hand=='S':
            b_hand = st.selectbox(f"Choose the player's hitting side:",['R','L'])
        pitcher_handednesses = list(timing_data.loc[(timing_data['pos']==pos_text) & (timing_data['id']==player_id) & (timing_data['bat_side']==b_hand),'pitch_hand'].value_counts().index)
        p_hand = st.selectbox(f"Choose the opposing pitchers' handedness:",pitcher_handednesses)
    
swing_count_dict, chart_df = transform_data(player_id,timing_data,pos_text,p_hand,b_hand)

if st.button('Generate Charts'):
    with st.spinner("Generating Charts...", show_time=True):
        miss_chart(chart_df,pos,b_hand,p_hand)
