import streamlit as st
import logic
import time

# --- CONFIGURATION ---
MAX_REGEX_LEN = 100
MAX_GEN_STRINGS = 1000

st.set_page_config(
    page_title="RegexLab // Pattern Compiler",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- BOLD RETRO-FUTURISTIC TERMINAL DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Chakra+Petch:wght@300;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Cyberpunk Terminal Theme */
    .stApp {
        background: #000000;
        color: #00ff41;
        font-family: 'JetBrains Mono', monospace;
        overflow-x: hidden;
    }
    
    /* Animated Scanlines */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(
            to bottom,
            transparent 50%,
            rgba(0, 255, 65, 0.02) 50%
        );
        background-size: 100% 4px;
        pointer-events: none;
        z-index: 1;
        animation: scanlines 0.5s linear infinite;
    }
    
    @keyframes scanlines {
        0% { transform: translateY(0); }
        100% { transform: translateY(4px); }
    }
    
    /* CRT Glow */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: radial-gradient(ellipse at center, transparent 0%, rgba(0, 0, 0, 0.4) 100%);
        pointer-events: none;
        z-index: 1;
    }
    
    /* Main Content Container */
    .main .block-container {
        padding: 2rem 4rem;
        position: relative;
        z-index: 2;
    }
    
    /* ASCII Art Header */
    .terminal-header {
        font-family: 'JetBrains Mono', monospace;
        color: #00ff41;
        text-shadow: 0 0 5px rgba(0, 255, 65, 0.3);
        font-size: 0.85rem;
        line-height: 1.2;
        margin-bottom: 2rem;
        white-space: pre;
        border: 2px solid #00ff41;
        padding: 1.5rem;
        background: rgba(0, 255, 65, 0.03);
        position: relative;
    }
    
    .terminal-header::before {
        content: "SYSTEM//ONLINE";
        position: absolute;
        top: -12px;
        right: 20px;
        background: #000000;
        padding: 0 10px;
        font-size: 0.7rem;
        color: #00ff41;
        letter-spacing: 2px;
    }
    
    /* Command Line Style Input */
    .stTextInput > label {
        font-family: 'Chakra Petch', sans-serif !important;
        color: #00ff41 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
    }
    
    .stTextInput > div > div > input {
        background: #0a0a0a !important;
        border: 2px solid #00ff41 !important;
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.1), 0 0 20px rgba(0, 255, 65, 0.2) !important;
        transition: all 0.3s ease !important;
        caret-color: #00ff41 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ff41 !important;
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.2), 0 0 40px rgba(0, 255, 65, 0.4) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(0, 255, 65, 0.3) !important;
    }
    
    /* Number Input Styling */
    .stNumberInput > label {
        font-family: 'Chakra Petch', sans-serif !important;
        color: #00ff41 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin-bottom: 0.3rem !important;
        font-weight: 600 !important;
    }
    
    .stNumberInput > div > div > input {
        background: #0a0a0a !important;
        border: 1px solid #00ff41 !important;
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        padding: 0.5rem !important;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.05) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #00ff41 !important;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.1), 0 0 10px rgba(0, 255, 65, 0.2) !important;
        outline: none !important;
    }
    
    /* Section Headers */
    .section-title {
        font-family: 'Chakra Petch', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff41;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin: 3rem 0 1.5rem 0;
        position: relative;
        padding-left: 20px;
    }
    
    .section-title::before {
        content: "▶";
        position: absolute;
        left: 0;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Terminal Box */
    .terminal-box {
        background: #0a0a0a;
        border: 2px solid #00ff41;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        box-shadow: inset 0 0 30px rgba(0, 255, 65, 0.05), 0 0 20px rgba(0, 255, 65, 0.1);
    }
    
    .terminal-box::before {
        content: attr(data-title);
        position: absolute;
        top: -12px;
        left: 15px;
        background: #000000;
        color: #00ff41;
        padding: 0 10px;
        font-size: 0.7rem;
        letter-spacing: 2px;
        font-family: 'Chakra Petch', sans-serif;
    }
    
    /* Graph Container */
    .graph-box {
        background: #ffffff;
        border: 2px solid #00ff41;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
    }
    
    /* Terminal Buttons */
    .stButton > button {
        background: #000000 !important;
        color: #00ff41 !important;
        border: 2px solid #00ff41 !important;
        font-family: 'Chakra Petch', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.8rem 2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 65, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: #00ff41 !important;
        color: #000000 !important;
        box-shadow: 0 0 40px rgba(0, 255, 65, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Code Output */
    .stCodeBlock {
        background: #0a0a0a !important;
        border: 1px solid #00ff41 !important;
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.05) !important;
    }
    
    .stCodeBlock code {
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Make code blocks scrollable */
    .stCodeBlock > div {
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    
    .stCodeBlock pre {
        max-height: 300px !important;
        overflow-y: auto !important;
        margin: 0 !important;
    }
    
    /* Scrollable Output Container */
    .output-scroll {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #00ff41;
        background: #0a0a0a;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .output-scroll::-webkit-scrollbar {
        width: 8px;
    }
    
    .output-scroll::-webkit-scrollbar-track {
        background: #0a0a0a;
        border-left: 1px solid #00ff41;
    }
    
    .output-scroll::-webkit-scrollbar-thumb {
        background: #00ff41;
        box-shadow: 0 0 5px #00ff41;
    }
    
    /* Status Messages */
    .stAlert {
        background: #0a0a0a !important;
        border: 2px solid #00ff41 !important;
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stSuccess {
        border-color: #00ff41 !important;
        background: rgba(0, 255, 65, 0.05) !important;
    }
    
    .stError {
        border-color: #ff0055 !important;
        background: rgba(255, 0, 85, 0.05) !important;
        color: #ff0055 !important;
    }
    
    .stWarning {
        border-color: #ffaa00 !important;
        background: rgba(255, 170, 0, 0.05) !important;
        color: #ffaa00 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00ff41 transparent transparent transparent !important;
    }
    
    /* Info Badge */
    .info-badge {
        display: inline-block;
        background: #0a0a0a;
        border: 1px solid #00ff41;
        color: #00ff41;
        padding: 0.5rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
        margin: 0.5rem 0;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    
    /* Stats Display */
    .stat-line {
        font-family: 'JetBrains Mono', monospace;
        color: #00ff41;
        font-size: 0.85rem;
        margin: 0.3rem 0;
        opacity: 0.8;
    }
    
    /* Footer Terminal */
    .footer-terminal {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #00ff41;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: rgba(0, 255, 65, 0.6);
        text-align: center;
    }
    
    /* Glitch Effect on Hover */
    .glitch {
        position: relative;
    }
    
    .glitch:hover {
        animation: glitch 0.3s infinite;
    }
    
    @keyframes glitch {
        0% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
        100% { transform: translate(0); }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #000000;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
        border: 1px solid #00ff41;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        box-shadow: 0 0 10px #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASCII HEADER ---
st.markdown("""
    <div class="terminal-header">
╔═══════════════════════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗ ██████╗ ███████╗██╗  ██╗    ██╗      █████╗ ██████╗    ║
║  ██╔══██╗██╔════╝██╔════╝ ██╔════╝╚██╗██╔╝    ██║     ██╔══██╗██╔══██╗   ║
║  ██████╔╝█████╗  ██║  ███╗█████╗   ╚███╔╝     ██║     ███████║██████╔╝   ║
║  ██╔══██╗██╔══╝  ██║   ██║██╔══╝   ██╔██╗     ██║     ██╔══██║██╔══██╗   ║
║  ██║  ██║███████╗╚██████╔╝███████╗██╔╝ ██╗    ███████╗██║  ██║██████╔╝   ║
║  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═════╝    ║
║                                                                            ║
║  PATTERN COMPILER V3.141592 // NEURAL GRAPH ENGINE // SYNTAX VALIDATOR    ║
╚═══════════════════════════════════════════════════════════════════════════╝
    </div>
    """, unsafe_allow_html=True)

# --- COMMAND INPUT ---
st.markdown("### PATTERN INPUT INTERFACE", unsafe_allow_html=True)

col_input, col_stats = st.columns([3, 1])

with col_input:
    regex_input = st.text_input(
        "► COMPILE REGEX PATTERN",
        value=r"([A-Z])\d{2,3}\1",
        help="Enter regex pattern for neural compilation"
    )

with col_stats:
    st.markdown(f"""
        <div style="padding-top: 2rem;">
            <div class="info-badge">
                MAX_LEN: {MAX_REGEX_LEN}
            </div>
            <div class="info-badge">
                MAX_GEN: {MAX_GEN_STRINGS}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN EXECUTION ---
if regex_input:
    if len(regex_input) > MAX_REGEX_LEN:
        st.error(f"⚠ PATTERN LENGTH EXCEEDS {MAX_REGEX_LEN} CHAR LIMIT")
    else:
        try:
            # Initialization
            with st.status("► INITIALIZING PATTERN COMPILER...", expanded=True) as status:
                time.sleep(0.2)
                builder = logic.RegexGraphBuilder(regex_input)
                status.update(label="✓ COMPILATION SUCCESS // GRAPH GENERATED", state="complete")
            
            # Display stats
            st.markdown(f"""
                <div class="stat-line">» Pattern Length: {len(regex_input)} chars</div>
                <div class="stat-line">» Engine Status: OPERATIONAL</div>
                <div class="stat-line">» Graph State: INITIALIZED</div>
                """, unsafe_allow_html=True)
            
            # Main Layout
            st.markdown('<div class="section-title">NEURAL TOPOLOGY GRAPH</div>', unsafe_allow_html=True)
            
            graph_col, control_col = st.columns([2, 1], gap="large")
            
            with graph_col:
                st.markdown('<div class="graph-box">', unsafe_allow_html=True)
                
                dot = builder.visualize()
                dot.attr(
                    bgcolor='white',
                    rankdir='LR',
                    margin='0.3',
                    pad='0.4',
                    ranksep='0.7',
                    nodesep='0.4'
                )
                dot.attr('node',
                        shape='box',
                        style='filled,rounded',
                        fillcolor='#f0fff0',
                        color='#00ff41',
                        fontcolor='#000000',
                        fontname='JetBrains Mono',
                        fontsize='10',
                        penwidth='2',
                        margin='0.2,0.1',
                        height='0.5')
                dot.attr('edge',
                        color='#00ff41',
                        arrowsize='0.8',
                        penwidth='2',
                        fontcolor='#000000',
                        fontname='JetBrains Mono',
                        fontsize='9')
                
                st.graphviz_chart(dot.source, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with control_col:
                # Positive Generation
                st.markdown('<div class="terminal-box" data-title="VALID STRINGS">', unsafe_allow_html=True)
                st.markdown("**GENERATE MATCHING PATTERNS**")
                st.markdown('<p style="color: rgba(0,255,65,0.6); font-size: 0.8rem; margin: 0.5rem 0 1rem 0;">Execute graph traversal to synthesize valid pattern matches</p>', unsafe_allow_html=True)
                
                num_positive = st.number_input(
                    "Number of samples",
                    min_value=1,
                    max_value=1000,
                    value=20,
                    step=1,
                    key="num_pos"
                )
                
                if st.button("⚡ EXECUTE", key="gen_pos"):
                    with st.spinner("► TRAVERSING GRAPH..."):
                        time.sleep(0.15)
                        pos_samples = builder.generate_sample(max_samples=MAX_GEN_STRINGS)
                        if pos_samples:
                            display_samples = pos_samples[:num_positive]
                            st.success(f"✓ GENERATED {len(pos_samples)} VALID STRINGS (SHOWING {len(display_samples)})")
                            st.text_area(
                                "Generated matches",
                                value="\n".join(display_samples),
                                height=320,
                                disabled=False,
                                label_visibility="collapsed",
                            )
                        else:
                            st.warning("⚠ NO VALID STRINGS FOUND")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Negative Generation
                st.markdown('<div class="terminal-box" data-title="VIOLATIONS">', unsafe_allow_html=True)
                st.markdown("**GENERATE VIOLATION CASES**")
                st.markdown('<p style="color: rgba(0,255,65,0.6); font-size: 0.8rem; margin: 0.5rem 0 1rem 0;">Synthesize negative test cases and edge violations</p>', unsafe_allow_html=True)
                
                num_negative = st.number_input(
                    "Number of violations",
                    min_value=1,
                    max_value=50,
                    value=15,
                    step=1,
                    key="num_neg"
                )
                
                if st.button("🔥 MUTATE", key="gen_neg"):
                    with st.spinner("► SIMULATING VIOLATIONS..."):
                        time.sleep(0.15)
                        neg_samples = builder.generate_negative_samples(num_samples=num_negative)
                        if neg_samples:
                            st.success(f"✓ GENERATED {len(neg_samples)} VIOLATIONS")
                            st.text_area(
                                "Generated violations",
                                value="\n".join(neg_samples),
                                height=320,
                                disabled=True,
                                label_visibility="collapsed",
                            )
                        else:
                            st.warning("⚠ NO VIOLATIONS SYNTHESIZED")
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"⚠ SYSTEM ERROR: {str(e)}")
            with st.expander("► VIEW ERROR LOG"):
                st.code(str(e), language="text")

# --- FOOTER ---
st.markdown("""
    <div class="footer-terminal">
    ───────────────────────────────────────────────────────────────────<br>
    REGEX_LAB v3.141592 // MAX_UNROLL=3 // RECURSION_DEPTH=0<br>
    SYSTEM STATUS: OPERATIONAL // UPTIME: ∞ // MEMORY: OPTIMAL<br>
    ───────────────────────────────────────────────────────────────────
    </div>
    """, unsafe_allow_html=True)
