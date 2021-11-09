phonem_mapping_str = '''aa	aa	aa
ae	ae	ae
ah	ah	ah
ao	ao	aa
aw	aw	aw
ax	ax	ah
ax-h	ax	ah
axr	er	er
ay	ay	ay
b	b	b
bcl	vcl	sil
ch	ch	ch
d	d	d
dcl	vcl	sil
dh	dh	dh
dx	dx	dx
eh	eh	eh
el	el	l
em	m	m
en	en	n
eng	ng	ng
epi	epi	sil
er	er	er
ey	ey	ey
f	f	f
g	g	g
gcl	vcl	sil
h#	sil	sil
hh	hh	hh
hv	hh	hh
ih	ih	ih
ix	ix	ih
iy	iy	iy
jh	jh	jh
k	k	k
kcl	cl	sil
l	l	l
m	m	m
n	n	n
ng	ng	ng
nx	n	n
ow	ow	ow
oy	oy	oy
p	p	p
pau	sil	sil
pcl	cl	sil
q	err	err
r	r	r
s	s	s
sh	sh	sh
t	t	t
tcl	cl	sil
th	th	th
uh	uh	uh
uw	uw	uw
ux	uw	uw
v	v	v
w	w	w
y	y	y
z	z	z
zh	zh	sh
spn	err	err
nsn	err	err
sp	sil	sil'''

phoneme_map_to_48 = {}
phoneme_map_to_39 = {}

mapping_lines = phonem_mapping_str.split('\n')
mapping_lines = [line.split() for line in mapping_lines]

for p1, p2, p3 in mapping_lines:
    phoneme_map_to_48[p1] = p2
    phoneme_map_to_39[p1] = p3
    phoneme_map_to_39[p2] = p3


def map_phoneme(phoneme, n_phonemes=39):
    assert n_phonemes in [60, 48, 39]
    if n_phonemes == 60:
        return phoneme
    if n_phonemes == 48:
        return phoneme_map_to_48[phoneme]
    if n_phonemes == 39:
        return phoneme_map_to_39[phoneme]


def get_phoneme_set(n_phonemes=39):
    assert n_phonemes in [60, 48, 39]
    phoneme_set = set()
    for p1, p2, p3 in mapping_lines:
        if n_phonemes == 60:
            phoneme_set.add(p1)
        if n_phonemes == 48:
            phoneme_set.add(p2)
        if n_phonemes == 39:
            phoneme_set.add(p3)

    return phoneme_set
