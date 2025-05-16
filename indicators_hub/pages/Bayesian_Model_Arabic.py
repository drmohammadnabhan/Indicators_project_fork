# In your content functions, for Arabic content:
ar_content_segments = [
    {"type": "html", "content": "هذه <b>مقدمة</b> مع بعض <i>التنسيق</i>."},
    {"type": "latex", "content": r"$$ E[\theta | D] = \frac{\alpha_{post}}{\alpha_{post} + \beta_{post}} $$"},
    {"type": "html", "content": "وهنا <b>المزيد</b> من النص."}
]
# English content can remain a single markdown string
en_content_md = "This is an **introduction** with some *formatting*.\n$$ E[\\theta | D] = \\frac{\\alpha_{post}}{\\alpha_{post} + \\beta_{post}} $$\nAnd here is **more** text."

# Modified display_markdown
def display_content(english_markdown, arabic_segments_list):
    if current_lang == "ar":
        # For Arabic, iterate through segments and apply RTL to the block
        st.markdown("<div dir='rtl'>", unsafe_allow_html=True)
        for segment in arabic_segments_list:
            if segment["type"] == "html":
                # Render HTML segment (already includes <b>, <i> etc)
                st.markdown(segment["content"], unsafe_allow_html=True)
            elif segment["type"] == "latex":
                # Render LaTeX segment using st.latex for KaTeX processing
                st.latex(segment["content"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # For English, render the single markdown string (which includes LaTeX)
        st.markdown(english_markdown)
