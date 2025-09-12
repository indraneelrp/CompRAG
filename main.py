'''
Caller of all helper functions
'''

# Grab queries
# Make triples from queries
# Do HRR for triples of the query
# Search k most similar triples for each of the query's triples
# Grab the associated chunks (do not duplicate grab)
# Pass it to the Ollama LM and let it produce its output
# save output for later eval of Hotpot QA