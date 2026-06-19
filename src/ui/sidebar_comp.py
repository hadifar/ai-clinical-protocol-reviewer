import streamlit as st

from ui import agent_view, ingest_view, search_view

footer = """<style>
a:link , a:visited{
color: blue;
text-decoration: none;
background-color: transparent;
}

a:hover,  a:active {
background-color: transparent;
}

.footer {
position: fixed;
bottom: 0;
color: black;
text-align: center;
font-size: smaller;
opacity: 0.9;
}
</style>
<div class="footer">
    <p>Developed with ❤️ by <a href="https://hadifar.github.io/" target="_blank">Amir Hadifar</a></p>
</div>
"""


def render():

    pages = [
        st.Page(
            ingest_view.render,
            title="Ingestion pipeline",
            icon="📥",
            url_path="ingestion",
        ),
        st.Page(
            search_view.render, title="Search pipeline", icon="🔍", url_path="search"
        ),
        st.Page(
            agent_view.render,
            title="IE agent",
            icon="🤖",
            url_path="agent",
        ),
    ]

    pg = st.navigation({"Main": pages})

    st.sidebar.markdown(footer, unsafe_allow_html=True)

    pg.run()
