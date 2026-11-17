# Références de petits pas conservant la trajectoire verticale relative

Outil de diagnostic **hors matériel**. Il ne modifie pas la téléop et ne lance
aucun contrôleur. Le fichier NPZ d'entrée contient les fenêtres du clip SMPL
apparié utilisées dans les études du 11 novembre (`pose`, `joints`, `quat`,
`wrists`, `indices`, `available_ns`). Les trames uniques doivent commencer à 0.

Depuis le checkout SONIC :

```sh
PYTHONPATH=. /home/ethan/Humanoid/GEAR-SONIC/.venv_teleop/bin/python \
  docs/measurements/smpl-main-2026-11-17/generate_short_stride.py \
  --reference ../smpl-wrist-fidelity-20261111/paired-producer-reference.npz \
  --output-dir /tmp/short-stride --amplitudes .25 1 --bend-plane forward
```

Par défaut, seule l'excursion X des chevilles est réduite ; leur Y/Z et
l'orientation des pieds sont conservés. Les longueurs des os sont conservées
par une construction géométrique, puis vérifiées par FK. Une cible hors de
portée ou un plan indéterminé échoue, sans modifier les longueurs des segments.
Les références existantes ne sont pas écrasées. Les options de centrage,
hauteur et plan de genou sont des variantes de diagnostic, pas des réglages
pour le robot. Le manifeste donne les empreintes de l'entrée et des sorties.

La hauteur est **relative au bassin**, pas mesurée au-dessus du sol humain.
La construction peut modifier la forme et la vitesse du genou ; une trajectoire
géométriquement correcte n'est pas certifiée dynamiquement réalisable.

Validation locale : les tableaux des variantes genoux vers l'avant à 25 % et
100 % reproduisent exactement ceux des essais MuJoCo. Le cas à plan original
100 % reproduit exactement les commandes, positions et racine du clip antérieur.
Une petite excursion X avec trajectoire Z conservée peut donner une excursion
robot excessive ; aucune amplification n'est donc activée en téléop.

Le rapport Controls `docs/smpl-main-comparison-2026-11-17.md`, sur
`low-latency-for-isaac-lab5`, contient les 15 rejeux, la comparaison des vrais
producteurs main/corrigé et les limites. Les symptômes physiques signalés
concernent main ; la branche corrigée n'a pas encore été testée sur le G1.
