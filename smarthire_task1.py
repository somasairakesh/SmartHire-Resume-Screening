from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
    "Python developer with machine learning skills",
    "Java developer with Spring Boot experience",
    "Frontend developer with React and JavaScript"
]

job_description = ["Looking for Python developer with ML knowledge"]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(resumes + job_description)

similarity = cosine_similarity(vectors[-1], vectors[:-1])

for i, score in enumerate(similarity[0]):
    print(f"Resume {i+1} Score: {score:.2f}")
