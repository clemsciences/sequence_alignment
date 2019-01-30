import lingpy as lp
from lingpy.algorithm import squareform

seqs = ["kona", "kvinne", "queen"]
msa = lp.Multiple(seqs)
msa.prog_align()

print(msa)

languages = ['Norwegian', 'Swedish', 'Icelandic', 'Dutch', 'English']

distances = squareform([0.5, 0.67, 0.8, 0.2, 0.4, 0.7, 0.6, 0.8, 0.8, 0.3])

tree = lp.neighbor(distances, languages)
print(tree)
tree = lp.Tree(tree)
print(tree.asciiArt())
