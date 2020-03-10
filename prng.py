LEHMER_0 = 0
LEHMER_1 = 1
LEHMER_2 = 2
LEHMER_3 = 3
LEHMER_4 = 4
LEHMER_5 = 5
MRG = 6
MIXED_CG = 7
ICG = 8
EICG = 9


def extended_gcd(aa, bb):
    lastremainder, remainder = abs(aa), abs(bb)
    x, lastx, y, lasty = 0, 1, 1, 0
    while remainder:
        lastremainder, (quotient, remainder) = remainder, divmod(
            lastremainder, remainder)
        x, lastx = lastx - quotient*x, x
        y, lasty = lasty - quotient*y, y
    return lastremainder, lastx * (-1 if aa < 0 else 1), lasty * (-1 if bb < 0 else 1)


def modinv(a, m):
	g, x, y = extended_gcd(a, m)
	if g != 1:
		raise ValueError
	return x % m


def prng(g=LEHMER_0, j=1, z=None):
    rnd = [{"a": 354623, "z": 179875, "m": 236481},  # Lehmer CG
           {"a": 11234, "z": 65489, "m": 236417},  # Lehmer CG
           {"a": 76982, "z": 17456, "m": 33651},  # Lehmer CG
           {"a": 152717, "z": 135623, "m": 210321},  # Lehmer CG
           {"a": 197331, "z": 172361, "m": 254129},  # Lehmer CG
           {"a": 48271, "z": 172361, "m": 2**31-1},  # Lehmer CG
           {"a": 1071064, "z": [135623, 172361], "m":2**31-19},  # MRG
           {"a": 6364136223846793005, "z": 172361, "m": 2**64},  # Mixed CG
           {"a": 197331, "z": 172361, "m": 2**31-1},  # ICG
           {"a": 172361, "z": [None, 0], "m":2**48-59}]  # EICG
    if g in [0, 1, 2, 3, 4, 5]:
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for _ in range(j):
            z = (a*z) % m
            yield z
    elif g == 6:  # MRG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for _ in range(j):
            z[0], z[1] = z[1], (a*z[1]+2113664*z[0]) % m
            yield z[0]
    elif g == 7:  # Mixed CG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for _ in range(j):
            z = (a*z+1) % m
            yield z
    elif g == 8:  # ICG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for _ in range(j):
            z = modinv(z, m)+1
            yield z
    elif g == 9:  # EICG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for _ in range(j):
            z[0], z[1] = modinv(z[1]+a, m), z[1]+1
            yield z[0]
