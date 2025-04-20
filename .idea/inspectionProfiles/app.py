import streamlit as st
import pandas as pd
import re
import string
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji

# Ensure nltk stopwords are available
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# Preprocess Function
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\u202F?[APap][Mm] -'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    if not messages or not dates:
        return None

    min_len = min(len(messages), len(dates))
    messages = messages[:min_len]
    dates = dates[:min_len]

    clean_dates = [d.replace('\u202F', '').replace('\u200e', '').strip() for d in dates]

    df = pd.DataFrame({'user_message': messages, 'message_date': clean_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M%p -', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Ensure the 'date' column is valid
    if df['date'].isnull().any():
        st.error("Some dates are not properly formatted!")

    df['user'] = df['user_message'].apply(
        lambda x: re.split(r'([^:]+):\s', x, maxsplit=1)[1] if ':' in x else 'group_notification')
    df['message'] = df['user_message'].apply(lambda x: re.split(r'([^:]+):\s', x, maxsplit=1)[2] if ':' in x else x)

    df.drop(columns=['user_message'], inplace=True)

    # Extract date-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_name'] = df['date'].dt.day_name()  # Ensure 'day_name' column is created

    return df


# Streamlit App
st.title("📱 WhatsApp Chat Analyzer")

uploaded_file = st.file_uploader("Upload your WhatsApp chat (.txt)", type="txt")

# Initialize df as None at the beginning to avoid 'df' not being defined error
df = None

if uploaded_file is not None:
    chat_data = uploaded_file.read().decode("utf-8")
    df = preprocess(chat_data)

    if df is not None and not df.empty:
        st.success("✅ Chat successfully processed.")
        st.subheader("📝 Sample Messages")
        st.dataframe(df.head(20))

# Dropdown to select user
if df is not None:
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")  # Option for total chat

    selected_user = st.selectbox("Select a user", user_list)

    # Layout: title and button
    title_col, button_col = st.columns([6, 1])

    with title_col:
        st.subheader("📊 Chat Summary")

    with button_col:
        show = st.button("Show Analysis", key="show_analysis_btn")

    if show:
        # Filter based on selected user
        if selected_user == "Overall":
            filtered_df = df
        else:
            filtered_df = df[df['user'] == selected_user]

        # --- Metrics ---
        total_messages = filtered_df.shape[0]
        total_words = filtered_df['message'].apply(lambda x: len(x.split())).sum()
        media_messages = filtered_df[filtered_df['message'] == '<Media omitted>'].shape[0]
        link_count = filtered_df['message'].str.contains('http').sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Total Words", total_words)
        with col3:
            st.metric("Media Shared", media_messages)
        with col4:
            st.metric("Links Shared", link_count)

        # --- Most Busy Users ---
        if selected_user == "Overall":
            st.subheader("👥 Most Busy Users")
            top_users = df['user'].value_counts().head(10)
            busy_df = pd.DataFrame({
                'User': top_users.index,
                'Message Count': top_users.values
            })
            st.bar_chart(busy_df.set_index('User'))

            # --- User Contribution Percentage ---
            st.subheader("📈 User Contribution Percentage")
            user_counts = df['user'].value_counts()
            user_percent = round((user_counts / user_counts.sum()) * 100, 2)

            percent_df = pd.DataFrame({
                'User': user_percent.index,
                'Contribution (%)': user_percent.values
            })

            st.bar_chart(percent_df.set_index('User'))
            st.dataframe(percent_df)

        # --- Messages Over Time ---
        st.subheader("📅 Messages Over Time")
        timeline = filtered_df.groupby(filtered_df['date'].dt.date).count()['message']
        st.line_chart(timeline)

        # --- WordCloud of Chat ---
        st.subheader("☁️ WordCloud of Most Used Words")

        # Join all messages into a single string
        text = " ".join(filtered_df['message'])

        # Create the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        # Display in Streamlit
        st.pyplot(fig)

        # --- Top 25 Most Used Words ---
        st.subheader("🔠 Top 25 Most Used Words")

        # Filter messages (remove media and system messages and blank rows)
        text_df = filtered_df[~filtered_df['message'].str.lower().str.contains('media omitted') &
                              ~filtered_df['message'].str.lower().str.contains('group_notification') &
                              (filtered_df['message'].str.strip() != '')]

        # Clean and tokenize words
        words = []
        for msg in text_df['message']:
            msg = msg.lower()
            msg = msg.translate(str.maketrans('', '', string.punctuation))
            for word in msg.split():
                if word not in stop_words and word.isalpha():  # Only keep alphabetic words
                    words.append(word)

        # Count frequency
        most_common_words = Counter(words).most_common(25)

        # Convert to DataFrame
        word_freq_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])

        # Display
        st.dataframe(word_freq_df)

        # --- Emoji Analysis ---
        st.subheader("🙂 Emoji Usage")
        emojis = []

        # Loop through all messages
        for message in filtered_df['message']:
            # Ignore media/system messages
            if '<media omitted>' not in message.lower() and 'group_notification' not in message.lower():
                for char in message:
                    if emoji.is_emoji(char):
                        emojis.append(char)


        # Count emoji frequency
        emoji_counter = Counter(emojis).most_common(10)
        emoji_df = pd.DataFrame(emoji_counter, columns=['Emoji', 'Count'])

        # --- Pie Chart of Top 5 Emojis ---
        top_5_emoji_df = emoji_df.head(5)

        # Create a pie chart
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust size of the pie chart

        # Set a custom color palette (adjust colors as needed)
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

        # Create the pie chart
        ax.pie(top_5_emoji_df['Count'], labels=top_5_emoji_df['Emoji'], autopct='%1.1f%%', startangle=140,
               colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
        ax.set_title("Top 5 Emojis Usage Share")

        # Adjust layout to prevent any cut-off
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

        # --- Monthly Timeline ---
        st.subheader("📅 Monthly Message Timeline")
        monthly_timeline = filtered_df.groupby([filtered_df['date'].dt.to_period("M")]).count()['message']
        monthly_timeline.index = monthly_timeline.index.to_timestamp()
        st.line_chart(monthly_timeline)

        # --- Daily Activity ---
        st.subheader("📆 Daily Activity")
        daily_timeline = filtered_df.groupby(filtered_df['date'].dt.date).count()['message']
        st.line_chart(daily_timeline)

        # --- Activity by Day of Week ---
        st.subheader("📊 Activity by Day of Week")
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_activity = filtered_df['day_name'].value_counts().reindex(dow_order)
        st.bar_chart(day_activity)

        # --- Hourly Activity ---
        st.subheader("⏰ Hourly Activity")
        hourly_activity = filtered_df['hour'].value_counts().sort_index()
        st.line_chart(hourly_activity)

        # --- Weekly Heatmap (Day vs. Hour) ---
        st.subheader("🔥 Weekly Activity Heatmap")
        heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(dow_order)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        ax.set_title("Messages by Day and Hour")
        st.pyplot(fig)

        # --- Day-based Activity per User ---
        st.subheader("📅 User Activity by Day")
        day_activity_user = filtered_df.groupby([filtered_df['user'], filtered_df['date'].dt.date]).size().unstack(
            fill_value=0)
        st.bar_chart(day_activity_user)

        # --- Month-based Activity per User ---
        st.subheader("📆 User Activity by Month")
        month_activity_user = filtered_df.groupby(
            [filtered_df['user'], filtered_df['date'].dt.to_period("M")]).size().unstack(fill_value=0)

        # Convert PeriodIndex to string format (e.g., '2025-04')
        month_activity_user.index = month_activity_user.index.astype(str)

        # Displaying the activity per user by month
        st.line_chart(month_activity_user)

        # --- Most Active Day per User ---
        most_active_day = day_activity_user.max(axis=1)
        st.subheader("📅 Most Active Day per User")
        st.write(most_active_day)

        # --- Most Active Month per User ---
        most_active_month = month_activity_user.max(axis=1)
        st.subheader("📆 Most Active Month per User")
        st.write(most_active_month)
