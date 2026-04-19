import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioAI — Indian Fund Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Syne:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #1E1E2F;
    color: #F8FAFC;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #F8FAFC !important;
}

.profile-card {
    background: #252537;
    border: 1px solid #2e2e45;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 12px 0;
}

.profile-conservative {
    border-left: 4px solid #3DDC97;
}
.profile-moderate {
    border-left: 4px solid #4F8CFF;
}
.profile-aggressive {
    border-left: 4px solid #FF6B6B;
}

.metric-box {
    background: #252537;
    border: 1px solid #2e2e45;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #3DDC97;
    font-family: 'Syne', sans-serif;
}

.metric-label {
    font-size: 0.78rem;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.fund-row {
    background: #252537;
    border: 1px solid #2e2e45;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.badge-aggressive { background: #FF6B6B20; color: #FF6B6B; border: 1px solid #FF6B6B40; }
.badge-moderate   { background: #4F8CFF20; color: #4F8CFF; border: 1px solid #4F8CFF40; }
.badge-conservative { background: #3DDC9720; color: #3DDC97; border: 1px solid #3DDC9740; }

.sidebar-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    color: #3DDC97;
    margin-bottom: 4px;
}

.stSlider > div > div > div { background: #4F8CFF !important; }

.stButton > button {
    background: linear-gradient(135deg, #3DDC97, #4F8CFF);
    color: #1E1E2F;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox label, .stRadio label, .stSlider label {
    color: #A0AEC0 !important;
    font-weight: 500;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #252537;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #A0AEC0;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #2e2e45;
    color: #F8FAFC !important;
}

div[data-testid="stDataFrame"] { border-radius: 10px; }

.section-divider {
    border: none;
    border-top: 1px solid #2e2e45;
    margin: 24px 0;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    color: #F8FAFC;
    line-height: 1.2;
}

.hero-sub {
    color: #A0AEC0;
    font-size: 1.05rem;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & CLUSTER DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_cluster(csv_path: str):
    df = pd.read_csv(csv_path)

    features = ['return_3y', 'return_5y', 'sharpe', 'std_dev', 'expense_ratio']
    df_model = df.dropna(subset=features).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[features])

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model['cluster'] = km.fit_predict(X_scaled)

    cluster_means = df_model.groupby('cluster')[features].mean()
    cluster_return_rank = cluster_means['return_3y'].rank()
    label_map = {}
    for cid, rank in cluster_return_rank.items():
        if rank == 1:
            label_map[cid] = 'Conservative'
        elif rank == 2:
            label_map[cid] = 'Moderate'
        else:
            label_map[cid] = 'Aggressive'
    df_model['risk_label'] = df_model['cluster'].map(label_map)

    sil = silhouette_score(X_scaled, df_model['cluster'])
    return df, df_model, km, scaler, X_scaled, cluster_means, label_map, sil


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ALLOCATION = {
    'Conservative': {'Debt MF': 45, 'Index Fund': 20, 'Gold': 15, 'Equity MF': 10, 'Fixed Income': 10},
    'Moderate':     {'Equity MF': 35, 'Debt MF': 25, 'Index Fund': 20, 'Gold': 15, 'Fixed Income': 5},
    'Aggressive':   {'Equity MF': 50, 'Index Fund': 25, 'Gold': 10, 'Debt MF': 10, 'Fixed Income': 5},
}

FIXED_INCOME = {
    'PPF (Public Provident Fund)':    {'return': '7.1%',  'lock_in': '15 years', 'risk': 'Sovereign'},
    'SBI Fixed Deposit (5Y)':         {'return': '6.5%',  'lock_in': '5 years',  'risk': 'Low'},
    'RBI Floating Rate Savings Bond': {'return': '8.05%', 'lock_in': '7 years',  'risk': 'Sovereign'},
}

QUESTIONS = [
    {
        'key': 'q1_age',
        'label': '1. What is your age group?',
        'options': {
            'Under 30 (long runway ahead)': 5,
            '30–40': 4,
            '41–50': 3,
            '51–60': 2,
            'Above 60 (near / in retirement)': 1,
        }
    },
    {
        'key': 'q2_goal',
        'label': '2. What is your primary investment goal?',
        'options': {
            'Long-term wealth creation (10+ years)': 5,
            'Retirement planning': 4,
            'Children\'s education / marriage': 3,
            'Buying a house or car': 2,
            'Emergency fund / short-term savings': 1,
        }
    },
    {
        'key': 'q3_reaction',
        'label': '3. If your portfolio drops 20% in a month, you would…',
        'options': {
            'Stay calm and hold — markets recover': 5,
            'Buy more at lower prices': 4,
            'Slightly worried but wait it out': 3,
            'Prefer stable returns over high ones': 2,
            'Exit immediately to stop losses': 1,
        }
    },
    {
        'key': 'q4_income',
        'label': '4. What is your monthly household income?',
        'options': {
            'Above ₹1 Lakh': 5,
            '₹50K–₹1 Lakh': 4,
            '₹20K–₹50K': 3,
            '₹10K–₹20K': 2,
            'Below ₹10K': 1,
        }
    },
    {
        'key': 'q5_experience',
        'label': '5. What is your investment experience?',
        'options': {
            'Expert — stocks, derivatives, PMS': 5,
            'Has Mutual Funds / SIPs': 4,
            'Has FD / PPF / NSC': 3,
            'Only savings account': 2,
            'Complete beginner': 1,
        }
    },
    {
        'key': 'q6_volatility',
        'label': '6. How comfortable are you with portfolio volatility?',
        'options': {
            'Completely comfortable — volatility = opportunity': 5,
            'Mostly comfortable': 4,
            'Somewhat okay with moderate swings': 3,
            'Prefer stability with small returns': 2,
            'Cannot tolerate any losses': 1,
        }
    },
]

PROFILE_COLORS = {'Conservative': '#3DDC97', 'Moderate': '#4F8CFF', 'Aggressive': '#FF6B6B'}
PROFILE_EMOJI = {'Conservative': '🛡️', 'Moderate': '⚖️', 'Aggressive': '🚀'}
ASSET_COLORS = ['#3DDC97', '#4F8CFF', '#A0AEC0', '#B983FF', '#FF6B6B']


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_risk_profile(answers: dict):
    total = sum(answers.values())
    if total <= 15:
        return total, 'Conservative'
    elif total <= 22:
        return total, 'Moderate'
    else:
        return total, 'Aggressive'


def recommend_portfolio(df_model, profile, investment_amount=100000, top_n=3):
    alloc = ALLOCATION[profile]
    recommendations = {}
    for asset_class, pct in alloc.items():
        amount = (pct / 100) * investment_amount
        if asset_class == 'Fixed Income':
            recommendations[asset_class] = {
                'allocation_pct': pct, 'amount_inr': round(amount), 'funds': FIXED_INCOME
            }
            continue
        subset = df_model[
            (df_model['asset_class'] == asset_class) &
            (df_model['risk_label'] == profile)
        ].copy()
        if len(subset) < top_n:
            subset = df_model[df_model['asset_class'] == asset_class].copy()
        top_funds = (subset
                     .sort_values('return_3y', ascending=False)
                     .head(top_n)[['fund_name', 'category', 'return_3y', 'return_5y', 'sharpe', 'expense_ratio']]
                     .reset_index(drop=True))
        top_funds.index += 1
        recommendations[asset_class] = {
            'allocation_pct': pct, 'amount_inr': round(amount), 'funds': top_funds
        }
    return recommendations


def simulate_growth(df_model, profile, investment=100000):
    alloc = ALLOCATION[profile]
    rows = []
    total_final = 0
    for asset_class, pct in alloc.items():
        amount = (pct / 100) * investment
        if asset_class == 'Fixed Income':
            avg_return = 7.0
        else:
            subset = df_model[df_model['asset_class'] == asset_class]
            avg_return = subset['return_3y'].mean() if len(subset) > 0 else 8.0
        final = amount * ((1 + avg_return / 100) ** 3)
        total_final += final
        rows.append({'Asset Class': asset_class, 'Invested (₹)': round(amount),
                     'Avg 3Y Return (%)': round(avg_return, 1), 'Value after 3Y (₹)': round(final)})
    return pd.DataFrame(rows), round(total_final)


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def dark_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#252537')
    ax.tick_params(colors='#A0AEC0')
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')
    ax.title.set_color('#F8FAFC')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e2e45')
    return fig, ax


def plot_allocation_pie(profile):
    alloc = ALLOCATION[profile]
    labels = list(alloc.keys())
    sizes = list(alloc.values())
    color = PROFILE_COLORS[profile]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#1E1E2F')

    wedge_colors = [ASSET_COLORS[i] for i in range(len(labels))]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=wedge_colors,
        autopct='%1.0f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='#1E1E2F', linewidth=2.5)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('#1E1E2F')
    for t in texts:
        t.set_fontsize(10)
        t.set_color('#F8FAFC')
    ax.set_title(f'{PROFILE_EMOJI[profile]} {profile} Allocation',
                 fontsize=13, fontweight='bold', color='#F8FAFC', pad=16)
    plt.tight_layout()
    return fig


def plot_elbow(X_scaled):
    inertias, silhouettes = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1E1E2F')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#252537')
        ax.tick_params(colors='#A0AEC0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e2e45')
        ax.xaxis.label.set_color('#A0AEC0')
        ax.yaxis.label.set_color('#A0AEC0')

    ax1.plot(K_range, inertias, 'o-', color='#3DDC97', linewidth=2, markersize=7)
    ax1.axvline(x=3, color='#FF6B6B', linestyle='--', linewidth=1.5, label='K=3 (chosen)')
    ax1.set_title('Elbow Method (WCSS)', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')

    ax2.plot(K_range, silhouettes, 's-', color='#4F8CFF', linewidth=2, markersize=7)
    ax2.axvline(x=3, color='#FF6B6B', linestyle='--', linewidth=1.5, label='K=3 (chosen)')
    ax2.set_title('Silhouette Score', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')

    plt.suptitle('Optimal K Selection for K-Means', fontsize=13, fontweight='bold', color='#F8FAFC', y=1.02)
    plt.tight_layout()
    return fig


def plot_pca_scatter(df_model, X_scaled):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=df_model.index)
    df_pca['risk_label'] = df_model['risk_label'].values

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#252537')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e2e45')
    ax.tick_params(colors='#A0AEC0')
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')

    for label, color in PROFILE_COLORS.items():
        mask = df_pca['risk_label'] == label
        ax.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                   c=color, label=label, alpha=0.6, s=30, edgecolors='none')

    ax.set_title('K-Means Clusters (PCA 2D Projection)', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    legend = ax.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0', title='Risk Label')
    legend.get_title().set_color('#A0AEC0')
    plt.tight_layout()
    return fig


def plot_return_distribution(df_model):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#252537')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e2e45')
    ax.tick_params(colors='#A0AEC0')
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')

    for cls, color in zip(df_model['asset_class'].unique(), ASSET_COLORS):
        data = df_model[df_model['asset_class'] == cls]['return_3y'].dropna()
        ax.hist(data, bins=25, alpha=0.65, color=color, label=cls, edgecolor='none')

    ax.set_title('3Y Return Distribution by Asset Class', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax.set_xlabel('3-Year Return (%)')
    ax.set_ylabel('Number of Funds')
    legend = ax.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')
    plt.tight_layout()
    return fig


def plot_growth_bar(growth_df, profile):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#252537')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e2e45')
    ax.tick_params(colors='#A0AEC0')
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')

    x = range(len(growth_df))
    bars_inv = ax.bar(x, growth_df['Invested (₹)'], color='#2e2e45', label='Invested', width=0.5)
    bars_fin = ax.bar(x, growth_df['Value after 3Y (₹)'] - growth_df['Invested (₹)'],
                      bottom=growth_df['Invested (₹)'],
                      color=PROFILE_COLORS[profile], alpha=0.85, label='Growth', width=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(growth_df['Asset Class'], rotation=20, ha='right', color='#A0AEC0', fontsize=9)
    ax.set_title(f'₹ Growth Simulation — {profile} Portfolio (3 Years)', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax.set_ylabel('Amount (₹)')
    legend = ax.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')
    plt.tight_layout()
    return fig


def plot_sharpe_scatter(df_model):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_facecolor('#252537')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e2e45')
    ax.tick_params(colors='#A0AEC0')
    ax.xaxis.label.set_color('#A0AEC0')
    ax.yaxis.label.set_color('#A0AEC0')

    for label, color in PROFILE_COLORS.items():
        sub = df_model[df_model['risk_label'] == label]
        ax.scatter(sub['expense_ratio'], sub['sharpe'], c=color, label=label,
                   alpha=0.55, s=25, edgecolors='none')
    ax.set_title('Expense Ratio vs Sharpe Ratio', fontsize=12, fontweight='bold', color='#F8FAFC')
    ax.set_xlabel('Expense Ratio (%)')
    ax.set_ylabel('Sharpe Ratio')
    legend = ax.legend(facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">📊 PortfolioAI</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#A0AEC0;font-size:0.85rem">Indian Mutual Fund Recommender · ML-Powered</p>', unsafe_allow_html=True)
    st.markdown("---")

    csv_path = st.text_input("CSV Path", value="master_portfolio.csv",
                             help="Path to master_portfolio.csv relative to where you run streamlit")
    investment_amount = st.number_input("Investment Amount (₹)", min_value=5000, max_value=10000000,
                                        value=100000, step=5000)
    top_n = st.slider("Top N funds per category", min_value=2, max_value=5, value=3)

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", ["🏠 Home", "📝 Questionnaire", "📈 Recommendation", "🔬 EDA Explorer", "🧠 Model Insights"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="color:#2e2e45;font-size:0.75rem">K-Means · K=3 · StandardScaler<br>Features: 3Y/5Y return, Sharpe, Std Dev, Expense Ratio</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df, df_model, km, scaler, X_scaled, cluster_means, label_map, sil_score = load_and_cluster(csv_path)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Could not load data from `{csv_path}`. Make sure the CSV is in the same directory as `app.py`.\n\nError: {e}")


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<h1 class="hero-title">Your Personalised<br>Indian Portfolio Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">K-Means clustering on 900+ real Indian mutual funds · Risk profiling via 6-question questionnaire · Allocation tailored to Conservative, Moderate, or Aggressive investors.</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if data_loaded:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Funds</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(df_model)}</div><div class="metric-label">Clustered Funds</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{df["asset_class"].nunique()}</div><div class="metric-label">Asset Classes</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{sil_score:.3f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="profile-card profile-conservative">
        <h3 style='margin:0 0 8px'>① Answer 6 Questions</h3>
        <p style='color:#A0AEC0;margin:0'>Your age, goal, income, experience and risk tolerance shape your investor profile.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="profile-card profile-moderate">
        <h3 style='margin:0 0 8px'>② Get Risk Profiled</h3>
        <p style='color:#A0AEC0;margin:0'>A score out of 30 maps you to Conservative · Moderate · Aggressive.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="profile-card profile-aggressive">
        <h3 style='margin:0 0 8px'>③ See Your Portfolio</h3>
        <p style='color:#A0AEC0;margin:0'>Asset allocation + top K-Means clustered funds + 3-year growth simulation.</p>
        </div>""", unsafe_allow_html=True)

    if data_loaded:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Asset Class Breakdown")
        ac = df['asset_class'].value_counts().reset_index()
        ac.columns = ['Asset Class', 'Fund Count']
        st.dataframe(ac, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE: QUESTIONNAIRE
# ─────────────────────────────────────────────
elif page == "📝 Questionnaire":
    st.markdown("## 📝 Investor Risk Questionnaire")
    st.markdown('<p style="color:#A0AEC0">Answer all 6 questions honestly. Your total score (6–30) determines your risk profile.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    answers = {}
    for q in QUESTIONS:
        st.markdown(f"**{q['label']}**")
        chosen = st.radio("", list(q['options'].keys()), key=q['key'], label_visibility='collapsed')
        answers[q['key']] = q['options'][chosen]
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🎯 Calculate My Risk Profile"):
        total, profile = get_risk_profile(answers)
        st.session_state['profile'] = profile
        st.session_state['score'] = total
        st.session_state['investment'] = investment_amount

        color = PROFILE_COLORS[profile]
        emoji = PROFILE_EMOJI[profile]
        css_class = f"profile-{profile.lower()}"

        st.markdown(f"""
        <div class="profile-card {css_class}" style="margin-top:20px">
            <h2 style="margin:0 0 6px;color:{color}">{emoji} {profile} Investor</h2>
            <p style="color:#A0AEC0;margin:0 0 14px">Score: <b style="color:#3DDC97">{total}/30</b></p>
            <p style="color:#F8FAFC;margin:0">
            {'🛡️ You prefer capital preservation. Debt-heavy portfolio with low volatility.' if profile == 'Conservative' else
             '⚖️ Balanced mix of growth and safety. Equity + Debt in equal measure.' if profile == 'Moderate' else
             '🚀 You seek maximum growth. High equity allocation, comfortable with swings.'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.info("➡️ Switch to **📈 Recommendation** in the sidebar to see your personalised fund picks.")


# ─────────────────────────────────────────────
# PAGE: RECOMMENDATION
# ─────────────────────────────────────────────
elif page == "📈 Recommendation":
    st.markdown("## 📈 Portfolio Recommendation")

    # Allow manual override if questionnaire not taken
    profile_options = ['Conservative', 'Moderate', 'Aggressive']
    saved_profile = st.session_state.get('profile', 'Moderate')
    saved_score = st.session_state.get('score', None)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        profile = st.selectbox("Risk Profile", profile_options, index=profile_options.index(saved_profile))
    with col_r:
        if saved_score:
            st.markdown(f"<br><span style='color:#A0AEC0'>Questionnaire score: <b style='color:#3DDC97'>{saved_score}/30</b></span>", unsafe_allow_html=True)

    if not data_loaded:
        st.warning("Please load data first.")
    else:
        reco = recommend_portfolio(df_model, profile, investment_amount, top_n)

        st.markdown(f"### {PROFILE_EMOJI[profile]} {profile} Portfolio — ₹{investment_amount:,.0f}")

        # Pie chart + allocation table side by side
        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            st.pyplot(plot_allocation_pie(profile))
        with col_tbl:
            alloc_rows = [{'Asset Class': k, 'Allocation %': v, 'Amount (₹)': f"₹{round((v/100)*investment_amount):,}"}
                          for k, v in ALLOCATION[profile].items()]
            st.dataframe(pd.DataFrame(alloc_rows), hide_index=True, use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Recommended Funds")

        for asset_class, details in reco.items():
            color = PROFILE_COLORS[profile]
            with st.expander(f"**{asset_class}** — {details['allocation_pct']}% · ₹{details['amount_inr']:,}", expanded=True):
                if asset_class == 'Fixed Income':
                    fi_rows = [{'Instrument': name, 'Return': info['return'],
                                'Lock-in': info['lock_in'], 'Risk': info['risk']}
                               for name, info in details['funds'].items()]
                    st.dataframe(pd.DataFrame(fi_rows), hide_index=True, use_container_width=True)
                else:
                    funds_df = details['funds'].copy()
                    funds_df.columns = ['Fund Name', 'Category', '3Y Return (%)', '5Y Return (%)', 'Sharpe', 'Expense Ratio (%)']
                    st.dataframe(funds_df, use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### 📈 3-Year Growth Simulation")
        growth_df, total_final = simulate_growth(df_model, profile, investment_amount)
        gain = total_final - investment_amount

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">₹{investment_amount:,.0f}</div><div class="metric-label">Invested</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">₹{total_final:,.0f}</div><div class="metric-label">Value after 3Y</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">+{gain/investment_amount*100:.1f}%</div><div class="metric-label">Total Gain</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_bar, col_sim = st.columns([1.4, 1])
        with col_bar:
            st.pyplot(plot_growth_bar(growth_df, profile))
        with col_sim:
            st.dataframe(growth_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────
elif page == "🔬 EDA Explorer":
    st.markdown("## 🔬 Exploratory Data Analysis")

    if not data_loaded:
        st.warning("Please load data first.")
    else:
        tabs = st.tabs(["Return Distribution", "Sharpe vs Expense", "Risk by Asset Class", "Summary Stats"])

        with tabs[0]:
            st.markdown("#### 3-Year Return Distribution by Asset Class")
            st.pyplot(plot_return_distribution(df_model))

        with tabs[1]:
            st.markdown("#### Expense Ratio vs Sharpe Ratio (by Risk Cluster)")
            st.pyplot(plot_sharpe_scatter(df_model))

        with tabs[2]:
            st.markdown("#### Risk Label Distribution by Asset Class")
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor('#1E1E2F')
            ax.set_facecolor('#252537')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2e2e45')
            ax.tick_params(colors='#A0AEC0')
            cross = pd.crosstab(df_model['asset_class'], df_model['risk_label'])
            cross.plot(kind='bar', ax=ax, color=['#FF6B6B', '#3DDC97', '#4F8CFF'],
                       edgecolor='#1E1E2F', linewidth=0.5, width=0.7)
            ax.set_title('Risk Label Distribution by Asset Class', fontsize=12, fontweight='bold', color='#F8FAFC')
            ax.set_xlabel('')
            ax.set_ylabel('Number of Funds', color='#A0AEC0')
            legend = ax.legend(title='Risk Label', facecolor='#252537', edgecolor='#2e2e45', labelcolor='#A0AEC0')
            legend.get_title().set_color('#A0AEC0')
            plt.xticks(rotation=0, color='#F8FAFC')
            plt.tight_layout()
            st.pyplot(fig)

        with tabs[3]:
            st.markdown("#### Dataset Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)
            st.markdown("#### Missing Value Count")
            missing = df.isnull().sum().reset_index()
            missing.columns = ['Column', 'Missing Values']
            st.dataframe(missing[missing['Missing Values'] > 0], hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────
elif page == "🧠 Model Insights":
    st.markdown("## 🧠 K-Means Model Insights")

    if not data_loaded:
        st.warning("Please load data first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">3</div><div class="metric-label">Clusters (K)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{sil_score:.3f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(df_model)}</div><div class="metric-label">Funds Clustered</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["Elbow Curve", "PCA Scatter", "Cluster Means"])

        with tabs[0]:
            st.markdown("#### Optimal K Selection — Elbow Method & Silhouette")
            st.pyplot(plot_elbow(X_scaled))
            st.info("K=3 was chosen as the elbow point, corresponding to Conservative / Moderate / Aggressive profiles.")

        with tabs[1]:
            st.markdown("#### Cluster Visualisation — PCA 2D Projection")
            st.pyplot(plot_pca_scatter(df_model, X_scaled))
            st.caption("PCA reduces 5 financial features to 2 principal components for visualisation.")

        with tabs[2]:
            st.markdown("#### Cluster Mean Feature Values")
            means_display = cluster_means.copy()
            means_display.index = [label_map[i] for i in means_display.index]
            means_display.index.name = 'Risk Profile'
            st.dataframe(means_display.round(2), use_container_width=True)

            st.markdown("#### Cluster Size Distribution")
            size_df = df_model['risk_label'].value_counts().reset_index()
            size_df.columns = ['Risk Profile', 'Number of Funds']
            st.dataframe(size_df, hide_index=True, use_container_width=True)