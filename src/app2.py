from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage
import sqlalchemy

load_dotenv()

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    try:
        db = SQLDatabase.from_uri(db_uri)
        db.get_table_info()  # Probar conexión y obtener el esquema
        return db
    except sqlalchemy.exc.ProgrammingError as e:
        st.error(f"Error en la configuración de la base de datos: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
    return None

def get_sql_chain(db):
    def get_schema(_):
        try:
            return db.get_table_info()
        except Exception as e:
            st.error(f"Error al obtener el esquema de la base de datos: {e}")
            return ""

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database. Based on the table schema below, write a SQL query that answers the user's question. Take the conversation history into account. Translate everything into Spanish.

        <SCHEMA>{schema}</SCHEMA>

        Conversation history: {chat_history}

        Write only the SQL query and nothing else. Do not wrap the SQL query in another text, not even backticks.

        For example:
        Question: How many films are there?
        SQL query: SELECT COUNT(*) FROM film;
        Question: Name 10 actors
        SQL query: SELECT first_name, last_name FROM actor LIMIT 10;

        Your turn:

        Question: {question}
        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def clean_sql_query(query: str) -> str:
    # Elimina caracteres de escape innecesarios
    return query.replace("\\", "")

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database. Based on the table schema below, question, sql query, and sql response, write a natural language response. Translate everything into Spanish.
        <SCHEMA>{schema}</SCHEMA>

        Conversation history: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    def run_query(query):
        clean_query = clean_sql_query(query)
        try:
            return db.run(clean_query)
        except sqlalchemy.exc.ProgrammingError as e:
            st.error(f"Error en la consulta SQL: {e}")
            return ""
        except Exception as e:
            st.error(f"Error inesperado al ejecutar la consulta: {e}")
            return ""
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain)
        .assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: run_query(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
    except Exception as e:
        st.error(f"Error inesperado en el procesamiento: {e}")
        return ""

def configure_streamlit():
    st.set_page_config(page_title='Habla con tus datos', page_icon=":speech_balloon:")
    st.title('Habla con tus datos')

    with st.sidebar:
        st.subheader('Configuración')
        st.write("Soy un chat inteligente que puede responder a tus preguntas sobre tus bases de datos")

        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="Admin123!", key="Password")
        st.text_input("Database", value="sakila", key="Database")
        
        if st.button("Conectar"):
            with st.spinner("Conectando a la base de datos..."):
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                if db:
                    st.session_state.db = db
                    st.success("Conexión exitosa")
                else:
                    st.error("Error en la conexión a la base de datos. Por favor verifica las credenciales y la configuración.")

def display_chat_history(chat_history):
    for message in chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

def handle_user_query(user_query):
    if user_query:
        with st.spinner("Procesando tu pregunta..."):
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)
            
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            with st.chat_message("AI"):
                st.markdown(response)
            
            st.session_state.chat_history.append(AIMessage(content=response))

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hola, ¿qué quieres saber sobre tus datos?")]

    configure_streamlit()
    
    display_chat_history(st.session_state.chat_history)

    user_query = st.chat_input("Hola, ¿te puedo ayudar en algo?")
    if user_query:
        handle_user_query(user_query)

if __name__ == "__main__":
    main()

