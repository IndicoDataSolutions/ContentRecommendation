from sklearn.datasets import fetch_20newsgroups
import indicoio

print "Downloading data set"
posts = fetch_20newsgroups().data
joblib.dump(posts, "data/posts.jl")

print "Tagging %d posts" % len(posts)
post_tags = indicoio.batch_text_tags(posts)
joblib.dump(post_tags, "data/post_tags.jl")
