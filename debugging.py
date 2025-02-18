import streamlit as st


def main():
    st.title("Debugging App")
    st.write("This is a test message.")

    if st.button("Click Me"):
        st.write("Button clicked!")


if __name__ == "__main__":
    main()
