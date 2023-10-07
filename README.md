# Sentiment Analaysis on ChatGPT-related Tweets

Goals: predict sentiment of tweets and get topic of each sentiment

Data source: <a href="https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023">Kaggle | Data Source</a>

Reference: <a href="https://towardsdatascience.com/%EF%B8%8F-sentiment-analysis-aspect-based-opinion-mining-72a75e8c8a6d">Lowri Williams Medium</a>

Author: <a href="www.linkedin.com/in/surya-abdillah">Surya Abdillah</a>

# Methodology
here are some process that needed for sentiment analysis 

<ol>
  <li>Data Preprocessing</li>
    <ol>
      <li>Lower case</li>
      <li>Removing emojis</li>
      <li>Removing mentions, hashtags, and links</li>
      <li>Removing punctuations</li>
      <li>Remove stopwords</li>
      <li>Stemming</li>
      <li>Lemmazitation</li>
    </ol>
  <li>Lexicon-Based Sentiment Analysis</li>
  <li>Topic Modeling</li>
  <ol>
    <li>Using BERTopic</li>
    <li>Using Latent Dirichlet Allocation (LDA)</li>
  </ol>
</ol>

# Result and Analysis

## Data Preprocessing

From processes that already mentioned in Methodology, here are example of text before and after preprocessing:

Process  | Text1 | Text2 | Text3 |
:---: | :--- | :--- | :--- |
Raw Data | @MecoleHardman4 Chat GPT says it’s 15. 😂 | AI muses: "In the court of life, we must all face the judge of destiny and the jury of our actions. ⚖️🔮 #OutOfContextAI #AILifeLessons #ChatGPT | @techAU @elonmusk @TheChiefNerd Walt Disney tried to warn us... #ChatGPT #AGI https://t.co/CE7wqOBv7Y |
Lower Case | @mecolehardman4 chat gpt says it’s 15. 😂 | ai muses: "in the court of life, we must all face the judge of destiny and the jury of our actions. ⚖️🔮 #outofcontextai #ailifelessons #chatgpt | @techau @elonmusk @thechiefnerd walt disney tried to warn us... #chatgpt #agi https://t.co/ce7wqobv7y |
Removing emojis | @mecolehardman4 chat gpt says it's 15. | ai muses: "in the court of life, we must all face the judge of destiny and the jury of our actions. #outofcontextai #ailifelessons #chatgpt | @techau @elonmusk @thechiefnerd walt disney tried to warn us... #chatgpt #agi https://t.co/ce7wqobv7y |
Removing mentions, hashtags, and links | chat gpt says it's 15. | ai muses: "in the court of life, we must all face the judge of destiny and the jury of our actions.    |    walt disney tried to warn us...    |
Removing punctuations | chat gpt says it s 15 | ai muses   in the court of life  we must all face the judge of destiny and the jury of our actions     |    walt disney tried to warn us       | 
Remove stopwords | chat gpt says   15 | ai muses   court  life  must  face  judge  destiny   jury   actions | walt disney tried  warn us |
Stemming | chat gpt say 15 | ai muse court life must face judg destini juri action | walt disney tri warn us |
Lemmazitation | chat gpt say 15 | ai muse court life must face judg destini juri action | walt disney tri warn u |

Note: Data that used as Text1, Text2, and Text3 are from here:

Number | Date | ID |
:---: | :--- | :---: |
Text1 | 2023-03-29 22:58:18+00:00 | 1641213218520481805 |
Text2 | 2023-03-29 22:57:52+00:00 | 1641213110915571715 | 
Text3 | 2023-03-29 22:53:43+00:00 | 1641212064705449985 | 

## Lexicon-Based Sentiment Analysis

This method use an prebuilt tool Lexicon-Based Sentiment Analysis from Valence Aware Dictionary and sEntiment Reasoner (VADER) that will assign sentiment polarity score to each word in a text, considering valence and intensity. Actually VADER also handle emoji, but in this research we will remove the emoji. We will use this command to do sentiment analysis:

```
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sent_score = sia.polarity_scores(df['content_clean'][0])
sent_score
```

This command will resulting an dictionary consist of 4 keys:
- neg: The proportion of negative sentiment in the text.
- neu: The proportion of neutral sentiment in the text.
- pos: The proportion of positive sentiment in the text.
- compound: The overall sentiment intensity score, which can be used to classify the text as positive, negative, or neutral.

From our dataset we get distribution of sentiments are:
- negative : 466,247
- neutral : 24,126
- positive : 9,663

Here is the bar plot:

![image](https://github.com/SuryaAbdillah/Cyberdas_Text-Mining_Sentiment-Analysis/assets/97737970/17577752-d711-4124-95b5-8110d1de4967)

### WordCloud (using top 30)
#### NEGATIVE

![image](https://github.com/SuryaAbdillah/Cyberdas_Text-Mining_Sentiment-Analysis/assets/97737970/fdd0005c-aba3-4987-a1ca-24e890944a0d)

#### NEUTRAL

![image](https://github.com/SuryaAbdillah/Cyberdas_Text-Mining_Sentiment-Analysis/assets/97737970/d85c439e-3fb6-4344-b2aa-f095f4c70276)

#### POSITIVE

![image](https://github.com/SuryaAbdillah/Cyberdas_Text-Mining_Sentiment-Analysis/assets/97737970/7748228e-3c90-49ba-a1ee-3a59c1810260)

## Topic Modelling

### Using BERTopic

We are using BERTopic with default parameter, and here is the result:

#### Negative Sentiment (237 Topics, included -1 [outlier])
| Topic | Count | Name | Representative | Representative_Docs |
| :---: | :--- | :--- | :--- | :--- | 
| 1 | 118 | 1_danger_lurk_navel_exponenti | ['danger', 'lurk', 'navel', 'exponenti', 'gaze', 'fals', 'view', 'career', 'rape', 'zone'] | ['danger', 'danger', 'danger'] |
| 2 | 101 | 2_shit_sick_holi_stress | ['shit', 'sick', 'holi', 'stress', 'fr', 'chat', 'gpt', 'footbal', 'bro', 'cri'] | ['chat gpt shit', 'chat shit gpt', 'chat gpt shit'] |
| 3 | 101 | 3_chatgpt_suck_crap_weak | ['chatgpt', 'suck', 'crap', 'weak', 'dick', 'poor', 'sell', 'bad', 'dumber', 'chatbox'] | ['chatgpt suck', 'chatgpt suck', 'chatgpt suck'] |

#### Neutral Sentiment (5031 Topics, included -1 [outlier])
| Topic | Count | Name | Representative | Representative_Docs |
| :---: | :--- | :--- | :--- | :--- | 
| 1 | 3107 | 1_song_music_lyric_rap | ['song', 'music', 'lyric', 'rap', 'album', 'eminem', 'playlist', 'chord', 'rapper', 'guitar'] | ['could use write song video test chatgpt write r b song', 'write song ai', 'ask creat song featur think lyric read like someth would make'] |
| 2 | 1758 | 2_robot_boston_humanoid_overlord	 | ['robot', 'boston', 'humanoid', 'overlord', 'dynam', 'upris', 'reinforc', 'autonom', 'nighycaf', 'control'] | ['chat gpt ask robot', 'robot', 'robot'] |
| 3 | 1587 | 3_classroom_teacher_lesson_curriculum | ['classroom', 'teacher', 'lesson', 'curriculum', 'teach', 'student', 'educ', 'class', 'pedagogi', 'ass'] | ['creativ classroom worri student use ai like classroom', 'would name tool like design specif classroom student teacher use would want support learn s', 'tri use classroom lesson plan'] |

#### Positive Sentiment (493 Topics, included -1 [outlier])
| Topic | Count | Name | Representative | Representative_Docs |
| :---: | :--- | :--- | :--- | :--- | 
| 1 | 182 | 1_chatgpt_london_song_place | ['chatgpt', 'london', 'song', 'place', 'fail', 'overflow', 'app', 'stack', 'popular', 'wrong'] | ['chatgpt best', 'chatgpt best', 'chatgpt best'] | 
| 2 | 165 | 2_hoodi_shirt_pre_hat | ['hoodi', 'shirt', 'pre', 'hat', 'transform', 'love', 'train', 'fan', 'gpt', 'chat'] | ['love chatgpt fan super cool gener pre train transform gpt shirt hoodi hat', 'love chatgpt fan super cool gener pre train transform gpt shirt hoodi hat', 'love chatgpt fan super cool gener pre train transform gpt shirt hoodi hat'] | 
| 3 | 162 | 3_thank_thanks_hahah_shoutout | ['thank', 'thanks', 'hahah', 'shoutout', 'concern', 'million', 'fyi', 'expert', 'xd', 'ever'] | ['thank', 'thank', 'thank'] | 

### Using LDA (Latent Dirichlet Allocation)

With LDA we did hyperparameter tuning for some parameters in this research we will try to extract 5 until 11 topics, then we count the coherence score and choose best parameter with highest coherence score for out final model

#### NEGATIVE

