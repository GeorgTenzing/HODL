
# --- Compact User Spec ---
sol_6547  = [
    # Encoder
    ("conv", 1, 1, "save1"),
    ("pool",),
    ("conv", 1, 2),
      ("drop", 0.25, "save2"),
    ("pool",),
    ("conv", 2, 4, "save3"),
    ("pool",),
    ("conv", 4, 8, "save4"),
    ("pool",),
    ("conv", 8, 16, "save5"),
    ("pool",),
    ("conv", 16, 32, "save6"),
    ("pool",),
    ("conv", 32, 64, "save7"),
    
    # Bottleneck
    ("pool",),
    ("bottleneck", 64, 128),
      ("drop", 0.5),

    # Decoder
    ("deconv", 128, 64, "save7"),
    ("deconv", 64, 32, "save6"),
    ("deconv", 32, 16, "save5"),
    ("deconv", 16, 8, "save4"),
    ("deconv", 8, 4, "save3"),
      ("drop", 0.25),
    ("deconv", 4, 2, "save2"),
    ("deconv", 2, 1, "save1"),

    # Output
    ("out", 1, 1)
]