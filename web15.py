import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):

    np.bool = bool

def setup_chinese_font():

    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")


        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False # 确保可以显示负号

# ==============================================================================
# 1. 项目名称和配置
# ==============================================================================
st.set_page_config(
    page_title="基于XGBoost算法的早发心梗后心衰中西医结合预测模型",
    page_icon="❤️", 
    layout="wide"
)

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


feature_names_display = [
    'Outcome_CHD_DM',     # 糖尿病
    'Outcome_feiyan',     # 肺部感染
    'Tachyarrhythmia',    # 快速性心律失常
    'TCM',                # 中药干预
    'qizhixueyu',         # 气滞血瘀 (FIXED: 小写)
    'yangxu',             # 阳虚 (FIXED: 小写)
    'xueyushuiting',      # 血瘀水停 (FIXED: 小写)
    'age',                # 年龄 (FIXED: 小写)
    'Pulse_rate',         # 心率
    'Hb',                 # 血红蛋白
    'SCr',                # 血清肌酐
    'BUN'                 # 血尿素氮
]

# 12个特征的中文名称
feature_names_cn = [
    '糖尿病', '肺部感染', '快速性心律失常', '中药干预',
    '气滞血瘀', '阳虚', '血瘀水停',
    '年龄', '心率', '血红蛋白', '血清肌酐', '血尿素氮'
]

# 用于英文键名到中文显示名的映射
feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典：键名已修改为小写
variable_descriptions = {
    'Outcome_CHD_DM': '是否有糖尿病（0=无，1=有）',
    'Outcome_feiyan': '是否有肺部感染（0=无，1=有）',
    'Tachyarrhythmia': '是否有快速性心律失常（0=无，1=有）',
    'TCM': '是否有中药干预（0=无，1=有）',
    'qizhixueyu': '是否有气滞血瘀证（0=无，1=有）', 
    'yangxu': '是否有阳虚证（0=无，1=有）',          
    'xueyushuiting': '是否有血瘀水停证（0=无，1=有）', 
    'age': '年龄（岁）',                             
    'Pulse_rate': '心率（次/分）',
    'Hb': '血红蛋白（g/L）',
    'SCr': '血清肌酐（μmol/L）',
    'BUN': '血尿素氮（mmol/L）'
}

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):

    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 尝试获取模型内部特征名
        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:

                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型，请检查文件路径和格式: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # ==============================================================================
    # 2. 侧边栏和主标题
    # ==============================================================================
    # 侧边栏标题
    st.sidebar.title("早发心梗后心衰中西医结合预测模型")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200) 

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # 系统说明

    ## 关于本系统
    这是一个基于XGBoost算法的**早发心肌梗死后心力衰竭**中西医结合预测系统，用于评估早发心梗患者发生心衰的风险。

    ## 预测结果
    系统输出：
    - **心力衰竭**发生概率
    - 未发生**心力衰竭**概率
    - 风险分层（低/中/高）
    """)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("变量说明"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict.get(feature, feature)}**: {variable_descriptions.get(feature, '无详细说明')}")


    # 主页面标题
    st.title("基于XGBoost算法的早发心梗后心衰中西医结合预测模型")
    st.markdown("### 请在下方录入全部特征后进行预测")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return


    # ==============================================================================
    # 3. 特征输入控件
    # ==============================================================================
    st.header("患者指标录入")
    # 使用 4 列布局来容纳 12 个特征
    col1, col2, col3, col4 = st.columns(4) 
    
    # 类别变量的格式化函数
    to_cn = lambda x: "有" if x == 1 else "无"

    # 输入控件
    # --- 第 1 列：西医合并症/干预 ---
    with col1:
        st.subheader("合并症/干预")
        # 糖尿病（0/1）
        outcome_chd_dm = st.selectbox("糖尿病", options=[0, 1], format_func=to_cn, index=0, key='dm') 
        # 肺部感染（0/1）
        outcome_feiyan = st.selectbox("肺部感染", options=[0, 1], format_func=to_cn, index=0, key='fy') 
        # 快速性心律失常（0/1）
        tachyarrhythmia = st.selectbox("快速性心律失常", options=[0, 1], format_func=to_cn, index=0, key='ta')
        # 中药干预（0/1）
        tcm = st.selectbox("中药干预", options=[0, 1], format_func=to_cn, index=0, key='tcm')

    # --- 第 2 列：中医证候 ---
    with col2:
        st.subheader("中医证候")
        # 气滞血瘀（0/1）
        qizhixueyu = st.selectbox("气滞血瘀证", options=[0, 1], format_func=to_cn, index=0, key='qzxy')
        # 阳虚（0/1）
        yangxu = st.selectbox("阳虚证", options=[0, 1], format_func=to_cn, index=0, key='yx')
        # 血瘀水停（0/1）
        xueyushuiting = st.selectbox("血瘀水停证", options=[0, 1], format_func=to_cn, index=0, key='xyst')
        # 占位符
        st.write("")

    # --- 第 3 列：基本生理指标 ---
    with col3:
        st.subheader("基本信息")
        # 年龄（数值）
        age = st.number_input("年龄（岁）", value=60, step=1, min_value=18, max_value=120, key='age_val') # 注意：key不能与变量名age重复
        # 心率（数值）
        pulse_rate = st.number_input("心率（次/分）", value=75, step=1, min_value=30, max_value=150, key='pr')
        # 血红蛋白（数值）
        hb = st.number_input("血红蛋白（g/L）", value=120.0, step=1.0, key='hb')
        # 占位符
        st.write("")

    # --- 第 4 列：生化指标 ---
    with col4:
        st.subheader("生化指标")
        # 血清肌酐（数值）
        scr = st.number_input("血清肌酐（μmol/L）", value=80.0, step=0.1, key='scr')
        # 血尿素氮（数值）
        bun = st.number_input("血尿素氮（mmol/L）", value=5.0, step=0.1, key='bun')
        # 占位符
        st.write("")
        st.write("")


    # 预测按钮
    predict_button = st.button("开始预测", type="primary")

    if predict_button:

        user_inputs = {
            'Outcome_CHD_DM': outcome_chd_dm,
            'Outcome_feiyan': outcome_feiyan,
            'Tachyarrhythmia': tachyarrhythmia,
            'TCM': tcm,
            'qizhixueyu': qizhixueyu,       # FIXED
            'yangxu': yangxu,               # FIXED
            'xueyushuiting': xueyushuiting, # FIXED
            'age': age,                     # FIXED
            'Pulse_rate': pulse_rate,
            'Hb': hb,
            'SCr': scr,
            'BUN': bun,
        }

        # 特征对齐逻辑
        if model_feature_names:

            alias_to_user_key = {f: f for f in feature_names_display}
            
            resolved_values = []
            missing_features = []
            for c in model_feature_names: # 遍历模型要求的特征名
                ui_key = alias_to_user_key.get(c, c) 
                val = user_inputs.get(ui_key, user_inputs.get(c, None)) 
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)

            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}。\n请核对特征名（注意大小写）。")
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return

            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:

            ordered_cols = feature_names_display
            input_df = pd.DataFrame([[user_inputs[c] for c in ordered_cols]], columns=ordered_cols)

        # 简单检查缺失
        if input_df.isnull().any().any():
            st.error("存在缺失的输入值，请完善后重试。")
            return

        # 进行预测（概率）
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                # 假设第1列为阴性（未发生），第2列为阳性（发生）
                if len(proba) == 2:
                    no_aki_prob = float(proba[0])
                    aki_prob = float(proba[1])
                else:
                    raise ValueError("predict_proba返回的维度异常")
            else:
                # 预测失败的退路，概率近似
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(input_df))
                    aki_prob = 1 / (1 + np.exp(-score))
                    no_aki_prob = 1 - aki_prob
                else:
                    pred = int(model.predict(input_df)[0])
                    aki_prob = float(pred)
                    no_aki_prob = 1 - aki_prob

            # 显示预测结果
            st.header("心力衰竭风险预测结果")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("未发生心力衰竭概率")
                st.progress(no_aki_prob) 
                st.write(f"{no_aki_prob:.2%}")
            with col2:
                st.subheader("心力衰竭发生概率")
                st.progress(aki_prob) 
                st.write(f"{aki_prob:.2%}")

            # 风险分层
            risk_level = "低风险" if aki_prob < 0.3 else ("中等风险" if aki_prob < 0.7 else "高风险")
            risk_color = "green" if aki_prob < 0.3 else ("orange" if aki_prob < 0.7 else "red")
            st.markdown(f"### 心力衰竭风险评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)

            # 模型解释
            st.write("---")
            st.subheader("模型解释（SHAP）")
            # ===== SHAP 解释开始 =====
            try:
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                # 处理SHAP值格式
                if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    shap_value = shap_values[0, :, 1]
                    expected_value = explainer.expected_value[1]
                elif isinstance(shap_values, list):
                    shap_value = np.array(shap_values[1][0])
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    shap_value = np.array(shap_values[0])
                    expected_value = explainer.expected_value

                current_features = list(input_df.columns)

                # SHAP瀑布图
                st.subheader("SHAP瀑布图")
                try:
                    fig_waterfall = plt.figure(figsize=(12, 8))

                    # 将类别/二值特征的取值映射为中文显示（用于左侧标签的“值 = 特征名”）
                    display_data = input_df.iloc[0].copy()
                    to_cn_map = {0: '否', 1: '是'}
                    try:
                        if 'Outcome_CHD_DM' in display_data.index:
                            display_data['Outcome_CHD_DM'] = to_cn_map.get(int(display_data['Outcome_CHD_DM']), display_data['Outcome_CHD_DM'])
                        if 'Outcome_feiyan' in display_data.index:
                            display_data['Outcome_feiyan'] = to_cn_map.get(int(display_data['Outcome_feiyan']), display_data['Outcome_feiyan'])
                        if 'Tachyarrhythmia' in display_data.index:
                            display_data['Tachyarrhythmia'] = to_cn_map.get(int(display_data['Tachyarrhythmia']), display_data['Tachyarrhythmia'])
                        if 'TCM' in display_data.index:
                            display_data['TCM'] = to_cn_map.get(int(display_data['TCM']), display_data['TCM'])
                        # FIXED: SHAP显示键名改为小写
                        if 'qizhixueyu' in display_data.index:
                            display_data['qizhixueyu'] = to_cn_map.get(int(display_data['qizhixueyu']), display_data['qizhixueyu'])
                        if 'yangxu' in display_data.index:
                            display_data['yangxu'] = to_cn_map.get(int(display_data['yangxu']), display_data['yangxu'])
                        if 'xueyushuiting' in display_data.index:
                            display_data['xueyushuiting'] = to_cn_map.get(int(display_data['xueyushuiting']), display_data['xueyushuiting'])
                        # FIXED: age
                        if 'age' in display_data.index:
                            display_data['age'] = str(display_data['age']) # 数值变量不需要映射，转为字符串
                    except Exception:
                        pass

                    # 尝试使用中文特征名
                    try:
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_value,
                                base_values=expected_value,
                                data=display_data.values,
                                feature_names=[feature_dict.get(f, f) for f in current_features]
                            ),
                            max_display=len(current_features),
                            show=False
                        )
                    except Exception:
                        st.warning("中文特征名显示失败，使用英文特征名")
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_value,
                                base_values=expected_value,
                                data=display_data.values,
                                feature_names=current_features
                            ),
                            max_display=len(current_features),
                            show=False
                        )

                    # 强制使用中文字体
                    import matplotlib.font_manager as fm
                    if chinese_font:
                        for ax in fig_waterfall.get_axes():
                            for text in ax.texts:
                                text.set_fontfamily(chinese_font)
                            for label in ax.get_yticklabels():
                                label.set_fontfamily(chinese_font)
                            ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                            ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)

                    plt.tight_layout()
                    st.pyplot(fig_waterfall)
                    plt.close(fig_waterfall)

                except Exception as e:
                    st.error(f"无法生成瀑布图: {str(e)}")

                # SHAP力图
                st.subheader("SHAP力图")
                try:
                    import streamlit.components.v1 as components
                    import matplotlib

                    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                    matplotlib.rcParams['axes.unicode_minus'] = False

                    # 力图使用映射后的显示数据
                    display_data_fp = input_df.iloc[0].copy()
                    to_cn_map_fp = {0: '否', 1: '是'}
                    try:
                        if 'Outcome_CHD_DM' in display_data_fp.index:
                            display_data_fp['Outcome_CHD_DM'] = to_cn_map_fp.get(int(display_data_fp['Outcome_CHD_DM']), display_data_fp['Outcome_CHD_DM'])
                        if 'Outcome_feiyan' in display_data_fp.index:
                            display_data_fp['Outcome_feiyan'] = to_cn_map_fp.get(int(display_data_fp['Outcome_feiyan']), display_data_fp['Outcome_feiyan'])
                        if 'Tachyarrhythmia' in display_data_fp.index:
                            display_data_fp['Tachyarrhythmia'] = to_cn_map_fp.get(int(display_data_fp['Tachyarrhythmia']), display_data_fp['Tachyarrhythmia'])
                        if 'TCM' in display_data_fp.index:
                            display_data_fp['TCM'] = to_cn_map_fp.get(int(display_data_fp['TCM']), display_data_fp['TCM'])
                        # FIXED: SHAP显示键名改为小写
                        if 'qizhixueyu' in display_data_fp.index:
                            display_data_fp['qizhixueyu'] = to_cn_map_fp.get(int(display_data_fp['qizhixueyu']), display_data_fp['qizhixueyu'])
                        if 'yangxu' in display_data_fp.index:
                            display_data_fp['yangxu'] = to_cn_map_fp.get(int(display_data_fp['yangxu']), display_data_fp['yangxu'])
                        if 'xueyushuiting' in display_data_fp.index:
                            display_data_fp['xueyushuiting'] = to_cn_map_fp.get(int(display_data_fp['xueyushuiting']), display_data_fp['xueyushuiting'])
                        # FIXED: age
                        if 'age' in display_data_fp.index:
                            display_data_fp['age'] = str(display_data_fp['age']) # 数值变量不需要映射，转为字符串
                    except Exception:
                        pass

                    force_plot = shap.force_plot(
                        expected_value,
                        shap_value,
                        display_data_fp,
                        feature_names=[feature_dict.get(f, f) for f in current_features]
                    )

                    # 嵌入HTML
                    shap_html = f"""
                    <head>
                        {shap.getjs()}
                        <style>
                            /* 调整样式以适应Streamlit */
                            body {{ margin: 0; padding: 20px 10px 40px 10px; overflow: visible; }}
                            .force-plot {{ margin: 20px 0 40px 0 !important; padding: 20px 0 40px 0 !important; }}
                            svg {{ margin: 20px 0 40px 0 !important; }}
                            .force-plot-container {{ min-height: 200px !important; padding-bottom: 50px !important; }}
                        </style>
                    </head>
                    <body>
                        <div class="force-plot-container">{force_plot.html()}</div>
                    </body>
                    """
                    components.html(shap_html, height=400, scrolling=False)
                except Exception as e:
                    st.error(f"无法生成HTML力图: {str(e)}")

            except Exception as e:
                st.error(f"无法生成SHAP解释: {str(e)}")

        except Exception as e:
            st.error(f"预测或结果展示失败: {str(e)}")

    # 版权或说明
    st.write("---")
    st.caption("本系统由机器学习模型驱动，旨在辅助临床决策。请在专业医生指导下使用预测结果。")

if __name__ == "__main__":
    main()