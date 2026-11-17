# Rééchantillonnage SMPL — 17 novembre 2026

Le rééchantillonnage utilise maintenant une SLERP exacte, vectorisée sur les
21 articulations, pour conserver une vitesse angulaire constante entre deux
mesures. La racine suit la même interpolation, avec gestion des quaternions de
signes opposés. La cinématique directe reste calculée après interpolation.
Les timestamps, indices, fenêtres, calibration et modèles appris ne changent pas.

L'ancienne interpolation normalisée de quaternions déformait la vitesse entre
les extrémités. À un quart d'une rotation synthétique de 120°, l'erreur était
2,204° ; la SLERP supprime cette erreur mathématique. Cette amplitude entre
mesures n'est pas un mouvement humain normal revendiqué ni une enveloppe validée.
Le microbenchmark CPU des 21 rotations est dans
[les mesures](measurements/smpl-2026-11-17/interpolation.json), avec son script.
Il ne mesure pas la latence de téléop complète.

Les 162 tests téléop passent. Sept nouveaux cas échouent sur l'ancienne version
et passent sur la nouvelle : vitesse angulaire constante, signe de quaternion,
rotations multiaxes comparées à `scipy.spatial.transform.Slerp`. Les tests de
cadence, de rattrapage borné et de cohérence FK continuent de passer.

Le diagnostic sur l'enregistrement PICO du 9 novembre limite fortement la
portée pratique de ce correctif pour les jambes : maximum de 0,0921° de
changement de rotation et 0,694 mm aux chevilles entre les deux interpolations.
Ce n'est pas une mesure de vérité anatomique et **le problème des petits pas
n'est pas résolu**. Des rejeux G1 à 25 % d'amplitude restent pieds chargés même
avec l'horizon natif de dix poses et ses vitesses futures. À 50 %, cet horizon
améliore certains indicateurs ; ces données futures ne sont pas disponibles en
live. Aucun gain, amplification de geste ou modèle expérimental n'est activé.

Le rapport Controls `docs/smpl-fidelity-2026-11-17.md`, sur
`low-latency-for-isaac-lab5`, contient les rejeux, les limites et la préparation
à l'essai physique. Les traces locales sont dans
`../smpl-horizon-20261117` depuis le checkout SONIC utilisé. Aucun essai physique
n'a été réalisé pour cette correction.
