import streamlit as st
import requests
import json

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000/chat_router"
st.set_page_config(page_title="Advanced RAG Chatbot (Streaming)", layout="centered")

# --- UI ---
st.title("🤖 Advanced RAG Chatbot (Streaming)")
st.caption("Now with Ollama, Streaming, and Dynamic Status!")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist for this message
        if "sources" in message and message["sources"]:
            st.info(f"Sources: {', '.join(message['sources'])}")

# --- Chat Input ---
if prompt := st.chat_input("Ask me anything..."):
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the streaming server and display response
    with st.chat_message("assistant"):
        
        # New: Placeholder for the dynamic status message
        status_message = st.empty()
        
        # Place the spinner wrapper around the process
        with st.spinner("Thinking..."):
            try:
                # Make the streaming request
                response = requests.post(
                    SERVER_URL, 
                    json={"query": prompt}, 
                    stream=True
                )
                response.raise_for_status()

                # Placeholders for the streaming content
                sources_placeholder = st.empty()
                answer_placeholder = st.empty()
                
                full_answer = ""
                sources = []
                # Initial status while waiting for the first packet
                current_status = "Please wait, this may take a while..." 

                # Iterate over the ndjson stream
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse the JSON object on each line
                            data = json.loads(line.decode('utf-8'))
                            event = data.get("event")
                            
                            # --- Handle Status updates ---
                            if event == "status":
                                # Update the status message placeholder
                                current_status = data.get("data", "...")
                                # Display the status message
                                status_message.caption(f"Status: ({current_status})")
                                continue # Move to next line, no content to append

                            # --- Handle Sources update ---
                            if event == "sources":
                                # We got the sources packet
                                sources = data.get("data", [])
                                if sources:
                                    sources_placeholder.info(f"Sources: {', '.join(sources)}")
                            
                            # --- Handle Token update ---
                            elif event == "token":
                                # We got a token for the answer
                                token = data.get("data", "")
                                full_answer += token
                                # Update the answer placeholder with the new token
                                answer_placeholder.markdown(full_answer + "▌") # Add cursor
                            
                            # --- Handle Error update ---
                            elif event == "error":
                                # An error occurred on the server
                                error_msg = data.get("data", "Unknown error")
                                st.error(f"Server Error: {error_msg}")
                                break

                        except json.JSONDecodeError:
                            print(f"Warning: Received invalid JSON line: {line}")
                        except Exception as e:
                            print(f"Error processing stream data: {e}")
                            break

                # Clean up the cursor and final status message after streaming is complete
                answer_placeholder.markdown(full_answer)
                status_message.empty()
                
                # Save the full response to session state
                bot_message = {"role": "assistant", "content": full_answer}
                if sources:
                    bot_message["sources"] = sources
                st.session_state.messages.append(bot_message)

            except requests.exceptions.RequestException as e:
                error_message = f"Error connecting to server: {e}"
                status_message.empty()
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                status_message.empty()
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})