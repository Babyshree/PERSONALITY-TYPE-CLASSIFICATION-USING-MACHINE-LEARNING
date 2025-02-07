﻿# Personality_Analysis
## Overview
This project aims to predict personality types based on text data using machine learning techniques. The dataset contains posts from various personality types based on the Myers-Briggs Type Indicator (MBTI). We explore the data, extract relevant features, visualize the relationships between features, and implement multiple machine learning models for prediction.

## Dataset
The dataset mbti_1.csv contains the following columns:
type: Personality type (e.g., INFJ, ENTP, etc.)
posts: Text posts from different users

## Features Extracted

words_per_comment: Average number of words per comment

http_per_comment: Average number of URLs per comment

music_per_comment: Average number of mentions of "music" per comment

question_per_comment: Average number of questions per comment

img_per_comment: Average number of image mentions per comment

excl_per_comment: Average number of exclamation marks per comment

ellipsis_per_comment: Average number of ellipses per comment

## Data Preprocessing
Converted personality types to binary values for each of the four axes: I-E, N-S, T-F, J-P.
Split the dataset into training and testing sets.

## PERSONALITY TYPE
Personality type across 4 axis:

Introversion (I) – Extroversion (E)

Intuition (N) – Sensing (S)

Thinking (T) – Feeling (F)

Judging (J) – Perceiving (P)
