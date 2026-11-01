**To:** Puneet Jindal · **Subject:** Labellerr take-home: Option B (pre-label + review flagging)

Hi Puneet,

I picked Option B: 100 COCO images (person/car/dog) pre-labelled with YOLOv8n and uploaded to
Labellerr through the SDK as model suggestions, with low-confidence images in a separate "low" bucket.
Result: a calibrated threshold (0.77) auto-accepts 32% of images at 90.6% precision and flags
68%; my rough estimate is about 66% less labelling time (50 → 17 min). The 95% target wasn't
reachable; the note explains why, along with some SDK feedback from building it.

Code + one-page note + SDK feedback: [GITHUB LINK]
Short walkthrough recording: [RECORDING LINK]

Thanks for the interesting assignment. Happy to walk through it.
Manav
