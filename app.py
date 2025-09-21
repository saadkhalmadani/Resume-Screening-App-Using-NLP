import streamlit as st
from PyPDF2 import PdfReader
from utils import (extract_entities, match_resume_to_job, analyze_keyword_density, 
                   recommend_missing_skills, detect_experience_level, get_detailed_score_breakdown,
                   generate_improvement_recommendations, get_industry_advice)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening App using NLP")

mode = st.radio("🛠️ Choose Mode", ["Single Resume Match", "Multi Resume Comparison"])
job_description = st.text_area("📌 Paste the Job Description:")

def display_skill_tags(entities):
    skills = [ent for ent, label in entities if label in ['SKILL', 'ORG', 'PERSON']]
    top_skills = skills[:10]
    if top_skills:
        st.markdown("#### 🔖 Extracted Tags:")
        tag_html = " ".join([
            f"<span style='background-color:#e0f7fa; color:#006064; padding:4px 8px; margin:3px; border-radius:10px; display:inline-block;'>{skill}</span>"
            for skill in top_skills
        ])
        st.markdown(tag_html, unsafe_allow_html=True)

if mode == "Single Resume Match":
    uploaded_file = st.file_uploader("📤 Upload a Resume (PDF)", type=["pdf"])

    if uploaded_file and job_description:
        # Extract resume text safely
        resume_text = ""
        for page in PdfReader(uploaded_file).pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text

        st.write("🧾 Extracted resume text preview (first 500 chars):")
        st.write(resume_text[:500])

        if not resume_text.strip():
            st.error("❗ Could not extract any text from the resume. Please try another file.")
        else:
            # Get detailed analysis
            score_breakdown = get_detailed_score_breakdown(resume_text, job_description)
            entities = extract_entities(resume_text)
            keyword_analysis = analyze_keyword_density(resume_text, job_description)
            missing_skills = recommend_missing_skills(resume_text, job_description)
            experience_level, exp_score = detect_experience_level(resume_text)
            
            # Main Score Display
            st.subheader("🎯 Overall Match Score")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Score gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = score_breakdown['overall_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Match Score (%)"},
                    delta = {'reference': 70},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}}))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, width="stretch")
            
            with col2:
                st.metric("Experience Level", experience_level.split('(')[0], f"{exp_score*100:.0f}% match")
                st.metric("Missing Skills", len(missing_skills), "to improve")
            
            with col3:
                st.metric("Semantic Similarity", f"{score_breakdown['similarity_score']:.2f}", "AI Analysis")
                st.metric("Keyword Match", f"{score_breakdown['keyword_match_score']:.2f}", "Density Score")
            
            # Detailed Score Breakdown
            st.subheader("📊 Detailed Score Breakdown")
            
            # Create radar chart for score breakdown
            categories = ['Semantic\nSimilarity', 'Keyword\nMatching', 'Skills\nCoverage', 'Experience\nLevel']
            values = [
                score_breakdown['similarity_score'],
                score_breakdown['keyword_match_score'],
                score_breakdown['skills_coverage'],
                score_breakdown['experience_score']
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Score Breakdown',
                line_color='rgb(32, 201, 151)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                height=400,
                title="Performance Across Different Criteria"
            )
            st.plotly_chart(fig_radar, width="stretch")
            
            # Progress bars for each component
            st.subheader("📈 Score Components")
            
            components = [
                ("🔍 Semantic Similarity", score_breakdown['similarity_score'], "How well the resume content matches the job semantically"),
                ("🔑 Keyword Matching", score_breakdown['keyword_match_score'], "Frequency of important keywords from job description"),
                ("🛠️ Skills Coverage", score_breakdown['skills_coverage'], "Percentage of required skills mentioned"),
                ("👨‍💼 Experience Level", score_breakdown['experience_score'], "Seniority level match for the position")
            ]
            
            for name, score, description in components:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{name}**")
                    st.progress(score)
                    st.caption(description)
                with col2:
                    st.metric("Score", f"{score:.2f}", f"{score*100:.0f}%")
                st.write("")

            # Keyword Analysis
            if keyword_analysis:
                st.subheader("🔍 Keyword Density Analysis")
                
                keyword_df = pd.DataFrame([
                    {
                        'Keyword': keyword,
                        'Job Frequency': data['job_frequency'],
                        'Resume Frequency': data['resume_frequency'],
                        'Match Score': data['density_score']
                    }
                    for keyword, data in keyword_analysis.items()
                ]).sort_values('Match Score', ascending=False)
                
                # Keyword density chart
                fig_keywords = px.bar(
                    keyword_df.head(10), 
                    x='Keyword', 
                    y='Match Score',
                    title="Top Keywords Match Analysis",
                    color='Match Score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_keywords, width="stretch")
                
                # Detailed keyword table
                st.dataframe(keyword_df.head(10))

            # Missing Skills Recommendations
            if missing_skills:
                st.subheader("💡 Recommended Skills to Add")
                st.write("These skills appear in the job description but are missing from your resume:")
                
                # Create skill recommendation cards
                cols = st.columns(3)
                for i, skill in enumerate(missing_skills[:9]):
                    with cols[i % 3]:
                        st.info(f"🎯 **{skill}**")
                
                if len(missing_skills) > 9:
                    with st.expander(f"View {len(missing_skills) - 9} more recommendations"):
                        for skill in missing_skills[9:]:
                            st.write(f"• {skill}")

            st.subheader("📋 Extracted Entities")
            for ent, label in entities:
                st.markdown(f"- **{label}**: {ent}")

            display_skill_tags(entities)
            
            # 🚀 NEW: Comprehensive Improvement Recommendations
            st.subheader("🚀 How to Improve Your Score")
            
            improvements = generate_improvement_recommendations(resume_text, job_description, score_breakdown)
            industry_advice = get_industry_advice(job_description)
            
            # Score improvement potential
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"**Current Score: {score_breakdown['overall_score']*100:.1f}%** | **Potential: {min(95, score_breakdown['overall_score']*100 + 25):.1f}%** with improvements")
            with col2:
                st.metric("Industry", industry_advice['industry'], "Detected")
            
            # High Priority Improvements
            if improvements['high_priority']:
                st.markdown("### 🔥 **High Priority Actions** (Biggest Impact)")
                for i, rec in enumerate(improvements['high_priority'], 1):
                    with st.expander(f"{i}. {rec['title']} - {rec['impact']}", expanded=True):
                        st.write(f"**Issue:** {rec['description']}")
                        st.write(f"**Action:** {rec['action']}")
                        if 'example' in rec:
                            st.code(rec['example'], language=None)
            
            # Medium Priority Improvements
            if improvements['medium_priority']:
                st.markdown("### ⚡ **Medium Priority Actions**")
                for i, rec in enumerate(improvements['medium_priority'], 1):
                    with st.expander(f"{i}. {rec['title']} - {rec['impact']}"):
                        st.write(f"**Issue:** {rec['description']}")
                        st.write(f"**Action:** {rec['action']}")
                        if 'example' in rec:
                            st.code(rec['example'], language=None)
            
            # Specific Examples and Templates
            if improvements['specific_examples']:
                st.markdown("### 📝 **Before & After Examples**")
                for example in improvements['specific_examples']:
                    with st.expander(f"✨ {example['category']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**❌ Before:**")
                            st.text(example['before'])
                        with col2:
                            st.markdown("**✅ After:**")
                            st.text(example['after'])
                        st.info(f"💡 **Why it works:** {example['why']}")
            
            # Keyword Usage Suggestions
            if improvements['keyword_suggestions']:
                st.markdown("### 🔑 **Keyword Optimization**")
                keyword_df = pd.DataFrame([
                    {
                        'Keyword': kw['keyword'].title(),
                        'Current Usage': kw['current_usage'],
                        'Suggested Usage': kw['suggested_usage'],
                        'Example Context': kw['context_ideas'][0] if kw['context_ideas'] else 'Use in job descriptions'
                    }
                    for kw in improvements['keyword_suggestions']
                ])
                st.dataframe(keyword_df, width="stretch")
            
            # Industry-Specific Advice
            st.markdown(f"### 🏭 **{industry_advice['industry']} Industry Tips**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Must-Have Skills:**")
                for skill in industry_advice['must_have_skills']:
                    st.write(f"• {skill}")
                    
                st.markdown("**⭐ Nice-to-Have Skills:**")
                for skill in industry_advice['nice_to_have_skills']:
                    st.write(f"• {skill}")
            
            with col2:
                st.markdown("**📈 Key Industry Trends:**")
                for trend in industry_advice['key_trends']:
                    st.write(f"• {trend}")
                    
                st.markdown("**📄 Format Tips:**")
                for tip in industry_advice['format_tips']:
                    st.write(f"• {tip}")
            
            # Low Priority Improvements
            if improvements['low_priority']:
                with st.expander("🔧 **Additional Polish & Low Priority Items**"):
                    for i, rec in enumerate(improvements['low_priority'], 1):
                        st.markdown(f"**{i}. {rec['title']}** - {rec['impact']}")
                        st.write(f"• {rec['description']}")
                        st.write(f"• Action: {rec['action']}")
                        if 'example' in rec:
                            st.caption(f"Example: {rec['example']}")
                        st.write("")
            
            # Action Plan Summary
            st.markdown("### 📋 **Your Action Plan**")
            
            action_plan = []
            if improvements['high_priority']:
                action_plan.extend([f"🔥 {rec['title']}" for rec in improvements['high_priority']])
            if improvements['medium_priority']:
                action_plan.extend([f"⚡ {rec['title']}" for rec in improvements['medium_priority'][:2]])
            
            if action_plan:
                st.markdown("**Complete these actions in order for maximum score improvement:**")
                for i, action in enumerate(action_plan, 1):
                    st.write(f"{i}. {action}")
            
            # Export improvement plan
            st.markdown("### 📥 **Export Your Improvement Plan**")
            
            improvement_report = f"""# Resume Improvement Plan
            
## Current Score Analysis
- Overall Score: {score_breakdown['overall_score']:.2f} ({score_breakdown['overall_score']*100:.1f}%)
- Industry: {industry_advice['industry']}
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Priority Actions
"""
            
            for i, rec in enumerate(improvements['high_priority'], 1):
                improvement_report += f"\n### {i}. {rec['title']} - {rec['impact']}\n"
                improvement_report += f"**Issue:** {rec['description']}\n"
                improvement_report += f"**Action:** {rec['action']}\n"
                if 'example' in rec:
                    improvement_report += f"**Example:** {rec['example']}\n"
            
            for i, rec in enumerate(improvements['medium_priority'], 1):
                improvement_report += f"\n### {len(improvements['high_priority']) + i}. {rec['title']} - {rec['impact']}\n"
                improvement_report += f"**Issue:** {rec['description']}\n"
                improvement_report += f"**Action:** {rec['action']}\n"
            
            improvement_report += f"\n## Industry Recommendations ({industry_advice['industry']})\n"
            improvement_report += f"**Must-have skills:** {', '.join(industry_advice['must_have_skills'])}\n"
            improvement_report += f"**Key trends:** {', '.join(industry_advice['key_trends'])}\n"
            
            st.download_button(
                "📄 Download Improvement Plan",
                data=improvement_report,
                file_name=f'resume_improvement_plan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
                mime='text/markdown'
            )


elif mode == "Multi Resume Comparison":
    uploaded_files = st.file_uploader("📤 Upload Multiple Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        results = []
        for uploaded_file in uploaded_files:
            resume_text = ""
            for page in PdfReader(uploaded_file).pages:
                page_text = page.extract_text()
                if page_text:
                    resume_text += page_text

            if not resume_text.strip():
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}. Skipping.")
                continue

            # Get detailed analysis for each resume
            score_breakdown = get_detailed_score_breakdown(resume_text, job_description)
            entities = extract_entities(resume_text)
            missing_skills = recommend_missing_skills(resume_text, job_description)
            experience_level, _ = detect_experience_level(resume_text)
            
            top_entities = ', '.join([ent for ent, label in entities if label in ['SKILL', 'ORG', 'PERSON']][:5])
            
            results.append({
                "Resume File": uploaded_file.name,
                "Overall Score": round(score_breakdown['overall_score'], 3),
                "Semantic Score": round(score_breakdown['similarity_score'], 3),
                "Keyword Score": round(score_breakdown['keyword_match_score'], 3),
                "Skills Coverage": round(score_breakdown['skills_coverage'], 3),
                "Experience Level": experience_level.split('(')[0],
                "Missing Skills": len(missing_skills),
                "Top Entities": top_entities
            })

        if results:
            df = pd.DataFrame(results).sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1  # Start ranking from 1
            
            st.subheader("🏆 Ranked Resume Matches")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Resumes", len(df))
            with col2:
                st.metric("Avg Overall Score", f"{df['Overall Score'].mean():.2f}")
            with col3:
                best_resume = df.iloc[0]['Resume File']
                st.metric("Best Match", best_resume.split('.')[0][:15] + "...")
            with col4:
                st.metric("Best Score", f"{df.iloc[0]['Overall Score']:.2f}")
            
            # Comparison chart
            fig_comparison = px.bar(
                df.head(10), 
                x='Resume File', 
                y='Overall Score',
                title="Resume Comparison - Overall Scores",
                color='Overall Score',
                color_continuous_scale='RdYlGn'
            )
            fig_comparison.update_xaxes(tickangle=45)
            st.plotly_chart(fig_comparison, width="stretch")
            
            # Detailed breakdown chart
            if len(df) > 1:
                st.subheader("📊 Detailed Score Comparison")
                
                # Prepare data for radar chart comparison (top 5 resumes)
                top_5 = df.head(5)
                
                fig_multi_radar = go.Figure()
                
                for idx, row in top_5.iterrows():
                    fig_multi_radar.add_trace(go.Scatterpolar(
                        r=[row['Semantic Score'], row['Keyword Score'], row['Skills Coverage'], row['Overall Score']],
                        theta=['Semantic<br>Similarity', 'Keyword<br>Matching', 'Skills<br>Coverage', 'Overall<br>Score'],
                        fill='toself',
                        name=row['Resume File'][:20],
                        opacity=0.7
                    ))
                
                fig_multi_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    height=500,
                    title="Top 5 Resumes - Multi-Criteria Comparison"
                )
                st.plotly_chart(fig_multi_radar, width="stretch")
            
            # Enhanced data table with styling
            st.subheader("📋 Detailed Results Table")
            
            # Add color coding with simple highlighting
            def color_code_score(val):
                if val >= 0.8:
                    return 'background-color: #90EE90'  # Light green
                elif val >= 0.6:
                    return 'background-color: #FFFF99'  # Light yellow
                else:
                    return 'background-color: #FFB6C1'  # Light pink
            
            # Style the dataframe with simpler approach
            styled_df = df.style.format({
                'Overall Score': '{:.3f}',
                'Semantic Score': '{:.3f}',
                'Keyword Score': '{:.3f}',
                'Skills Coverage': '{:.3f}'
            }).map(color_code_score, subset=['Overall Score', 'Semantic Score', 'Keyword Score', 'Skills Coverage'])
            
            st.dataframe(styled_df, width="stretch")
            
            # Export functionality
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=True, index_label='Rank').encode('utf-8')
                st.download_button(
                    "� Download Detailed CSV Report", 
                    data=csv, 
                    file_name=f'resume_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 
                    mime='text/csv'
                )
            
            with col2:
                # Create summary report
                summary_report = f"""
                # Resume Screening Report
                **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                **Total Resumes Analyzed:** {len(df)}
                **Average Overall Score:** {df['Overall Score'].mean():.3f}
                
                ## Top 3 Candidates:
                """
                for i, row in df.head(3).iterrows():
                    summary_report += f"\n{i}. **{row['Resume File']}** - Score: {row['Overall Score']:.3f}"
                
                st.download_button(
                    "📄 Download Summary Report", 
                    data=summary_report, 
                    file_name=f'resume_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md', 
                    mime='text/markdown'
                )
                
        else:
            st.error("❗ No valid resumes processed. Please upload valid PDF resumes.")

else:
    st.info("Please upload resume(s) and enter the job description to get started.")
