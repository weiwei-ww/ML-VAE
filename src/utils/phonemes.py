phoneme_mapping_str = '''aa	aa	aa
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
sp	sil	sil
sil sil sil'''

phoneme_map_to_48 = {}
phoneme_map_to_39 = {}

mapping_lines = phoneme_mapping_str.split('\n')
mapping_lines = [line.split() for line in mapping_lines]

for p1, p2, p3 in mapping_lines:
    phoneme_map_to_48[p1] = p2
    phoneme_map_to_39[p1] = p3
    phoneme_map_to_39[p2] = p3


def get_phoneme_set(language='english', n_phonemes=39, **kwargs):
    if language.lower() == 'english':
        assert n_phonemes in [60, 48, 39]
        phoneme_set = []
        for p1, p2, p3 in mapping_lines:
            p = None
            if n_phonemes == 60:
                p = p1
            elif n_phonemes == 48:
                p = p2
            elif n_phonemes == 39:
                p = p3
            if p not in phoneme_set:
                phoneme_set.append(p)
    elif language.lower() == 'digits':
        phoneme_set = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'sil', 'err']
        assert n_phonemes == len(phoneme_set)
    elif language.lower() == 'pinyin':
        with open('utils/pinyin_dict.txt') as f:
            phoneme_set = [l.rstrip() for l in f.readlines()]
        assert n_phonemes == len(phoneme_set)
    else:
        raise ValueError(f'unknown language: {language}')

    return phoneme_set


class PhonemeSetHandler:
    def __init__(self, language='english', n_phonemes=39, **kwargs):
        if language == 'english':
            assert n_phonemes in [60, 48, 39]
        elif language == 'digits':
            assert n_phonemes in [11, 12]
        self.n_phonemes = n_phonemes
        self.phoneme_set = get_phoneme_set(language, n_phonemes, **kwargs)

    def get_phoneme_set(self):
        return self.phoneme_set

    def map_phoneme(self, p):
        if self.n_phonemes == 60 or p not in phoneme_map_to_48:
            return p
        if self.n_phonemes == 48:
            return phoneme_map_to_48[p]
        if self.n_phonemes == 39:
            return phoneme_map_to_39[p]
