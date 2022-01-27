from hashlib import sha1


def compute_hash(email):
    return sha1(email.lower().encode('utf-8')).hexdigest()
