import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# 대시보드 함수 - 기존 앱에 통합하기 위해 별도 함수로 생성
def render_document_correction_dashboard():
    st.header("문서교정 통계 대시보드", divider="red")
    
    # 탭 생성으로 다양한 뷰 제공
    tab1, tab2 = st.tabs(["교정 현황 개요", "상세 분석"])
    
    # 샘플 데이터 생성 (실제 앱에서는 DB나 파일에서 불러오기)
    def generate_sample_data():
        # 문서 유형 목록
        doc_types = ["계약서", "보고서", "제안서", "매뉴얼", "정책문서", "회의록", "법률문서", "기술문서"]
        
        # 오늘 날짜 기준 지난 30일 날짜 생성
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
        
        # 랜덤 데이터 생성
        data = []
        for date in dates:
            for doc_type in doc_types:
                # 각 문서 유형별 교정 수량 (랜덤)
                correction_count = np.random.randint(1, 15)
                
                # 오류 유형별 수량 (랜덤)
                grammar_errors = np.random.randint(1, 10)
                spelling_errors = np.random.randint(1, 8)
                style_issues = np.random.randint(0, 5)
                formatting_issues = np.random.randint(0, 6)
                
                # 교정 상태
                status = np.random.choice(["완료", "진행중", "대기중"], p=[0.7, 0.2, 0.1])
                
                # 교정 시간 (분 단위, 랜덤)
                correction_time = np.random.randint(10, 120)
                
                data.append({
                    "날짜": date,
                    "문서유형": doc_type,
                    "교정수량": correction_count,
                    "문법오류": grammar_errors,
                    "맞춤법오류": spelling_errors,
                    "문체오류": style_issues,
                    "서식오류": formatting_issues,
                    "상태": status,
                    "교정시간(분)": correction_time,
                    "총오류수": grammar_errors + spelling_errors + style_issues + formatting_issues,
                })
        
        return pd.DataFrame(data)
    
    # 샘플 데이터 로드
    df = generate_sample_data()
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 1행: 주요 지표 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = df["교정수량"].sum()
            st.metric("총 교정 문서", f"{total_docs:,}건")
        
        with col2:
            total_errors = df["총오류수"].sum()
            st.metric("총 발견 오류", f"{total_errors:,}건")
        
        with col3:
            avg_errors = round(df["총오류수"].sum() / df["교정수량"].sum(), 2)
            st.metric("문서당 평균 오류", f"{avg_errors}건")
        
        with col4:
            avg_time = round(df["교정시간(분)"].mean(), 1)
            st.metric("평균 교정 시간", f"{avg_time}분")
        
        # 2행: 중요 그래프 - 문서 유형별 현황
        st.subheader("문서 유형별 교정 현황")
        
        # 문서 유형별 교정 수량 집계
        doc_type_counts = df.groupby("문서유형").agg({
            "교정수량": "sum",
            "총오류수": "sum"
        }).reset_index()
        
        # 교정 수량 기준 정렬
        doc_type_counts = doc_type_counts.sort_values("교정수량", ascending=False)
        
        # 그래프 생성
        fig = px.bar(
            doc_type_counts,
            x="문서유형",
            y="교정수량",
            color="총오류수",
            color_continuous_scale="Reds",
            title="문서 유형별 교정 수량 및 오류 수",
            text_auto=True
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="문서 유형",
            yaxis_title="교정 수량 (건)",
            coloraxis_colorbar_title="총 오류 수"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3행: 오류 유형별 분포 (파이 차트)
        st.subheader("오류 유형별 분포")
        
        # 오류 유형별 합계
        error_types = {
            "문법 오류": df["문법오류"].sum(),
            "맞춤법 오류": df["맞춤법오류"].sum(),
            "문체 문제": df["문체오류"].sum(),
            "서식 문제": df["서식오류"].sum(),
        }
        
        error_df = pd.DataFrame({
            "오류 유형": list(error_types.keys()),
            "오류 수": list(error_types.values())
        })
        
        # 파이 차트 생성
        fig_pie = px.pie(
            error_df,
            names="오류 유형",
            values="오류 수",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        
        fig_pie.update_layout(
            height=400,
            legend_title="오류 유형"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 오류 유형별 비율 계산
            total = sum(error_types.values())
            st.markdown("### 오류 유형별 비율")
            
            for error_type, count in error_types.items():
                percentage = round((count / total) * 100, 1)
                st.write(f"**{error_type}**: {count:,}건 ({percentage}%)")
        
        # 4행: 교정 진행 상태 시각화
        st.subheader("교정 진행 상태")
        
        # 상태별 집계
        status_counts = df["상태"].value_counts().reset_index()
        status_counts.columns = ["상태", "문서수"]
        
        # 가로 막대 차트
        fig_status = px.bar(
            status_counts,
            x="문서수",
            y="상태",
            color="상태",
            color_discrete_map={
                "완료": "#4CAF50",
                "진행중": "#2196F3",
                "대기중": "#F44336"
            },
            orientation="h",
            text_auto=True
        )
        
        fig_status.update_layout(
            height=300,
            xaxis_title="문서 수",
            yaxis_title="상태",
            showlegend=False
        )
        
        st.plotly_chart(fig_status, use_container_width=True)
        
    # 탭 2: 상세 분석
    with tab2:
        st.subheader("문서 교정 상세 데이터")
        
        # 필터 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_doc_types = st.multiselect(
                "문서 유형 선택",
                options=sorted(df["문서유형"].unique()),
                default=sorted(df["문서유형"].unique())
            )
        
        with col2:
            date_range = st.date_input(
                "기간 선택",
                value=(
                    datetime.strptime(min(df["날짜"]), "%Y-%m-%d").date(),
                    datetime.strptime(max(df["날짜"]), "%Y-%m-%d").date()
                ),
                format="YYYY-MM-DD"
            )
        
        with col3:
            status_filter = st.multiselect(
                "상태 선택",
                options=sorted(df["상태"].unique()),
                default=sorted(df["상태"].unique())
            )
        
        # 필터 적용
        filtered_df = df.copy()
        
        if selected_doc_types:
            filtered_df = filtered_df[filtered_df["문서유형"].isin(selected_doc_types)]
        
        if len(date_range) == 2:
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
            filtered_df = filtered_df[(filtered_df["날짜"] >= start_date) & (filtered_df["날짜"] <= end_date)]
        
        if status_filter:
            filtered_df = filtered_df[filtered_df["상태"].isin(status_filter)]
        
        # 시간별 추이 그래프
        st.subheader("날짜별 교정 추이")
        
        # 날짜별 집계
        daily_data = filtered_df.groupby("날짜").agg({
            "교정수량": "sum",
            "총오류수": "sum",
            "교정시간(분)": "mean"
        }).reset_index()
        
        # 날짜 형식 변환
        daily_data["날짜"] = pd.to_datetime(daily_data["날짜"])
        daily_data = daily_data.sort_values("날짜")
        
        # 추이 그래프 (라인 차트)
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=daily_data["날짜"],
            y=daily_data["교정수량"],
            name="교정 수량",
            line=dict(color="#1E88E5", width=3)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=daily_data["날짜"],
            y=daily_data["총오류수"],
            name="오류 수",
            line=dict(color="#E53935", width=3)
        ))
        
        fig_trend.update_layout(
            height=400,
            xaxis_title="날짜",
            yaxis_title="건수",
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 교정 소요 시간 분석
        st.subheader("문서 유형별 평균 교정 시간")
        
        # 문서 유형별 평균 교정 시간
        avg_time_by_type = filtered_df.groupby("문서유형")["교정시간(분)"].mean().reset_index()
        avg_time_by_type = avg_time_by_type.sort_values("교정시간(분)", ascending=False)
        
        # 막대 차트
        fig_time = px.bar(
            avg_time_by_type,
            x="문서유형",
            y="교정시간(분)",
            color="교정시간(분)",
            color_continuous_scale="Viridis",
            text_auto='.1f'
        )
        
        fig_time.update_layout(
            height=400,
            xaxis_title="문서 유형",
            yaxis_title="평균 교정 시간 (분)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # 상세 데이터 테이블
        st.subheader("상세 데이터")
        
        # 필터링된 데이터 표시 (일부 열만)
        display_columns = ["날짜", "문서유형", "교정수량", "총오류수", "문법오류", "맞춤법오류", "문체오류", "서식오류", "상태", "교정시간(분)"]
        st.dataframe(filtered_df[display_columns], use_container_width=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_df[display_columns].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="데이터 CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv"
        )

# 독립 실행을 위한 코드 (테스트용)
if __name__ == "__main__":
    st.set_page_config(
        page_title="문서교정 대시보드",
        page_icon="📊",
        layout="wide"
    )
    render_document_correction_dashboard() 