flowchart TD
    A[Input Citra] --> B[Preprocessing]
    B --> C[Dilated U-Net Encoder]
    C --> D[Bottleneck Feature]
    D --> E[Spatial Attention]

    E --> F[Segmentation Decoder]
    F --> G[Mask Lesi]
    G --> H[Fitur Morfologi<br/>area, jumlah, distribusi]

    E --> I[Fitur Global]

    H --> J[AMFM Fusion]
    I --> J

    J --> K[Classification Head]
    K --> L[Severity Hayashi]


flowchart TD
    A[Input Citra Wajah] --> B[Preprocessing<br/>Resize + Normalisasi]
    B --> C[ResNet-50 Encoder<br/>Shared Feature Extractor]

    C --> D[Feature Map / Bottleneck]
    D --> E[Spatial Attention<br/>Fokus ke area lesi]

    E --> F[Dilated U-Net Decoder]
    F --> G[Mask Segmentasi Lesi]

    G --> H[Ekstraksi Fitur Morfologi<br/>jumlah, area, distribusi]

    E --> I[Fitur Global Wajah]

    H --> J[AMFM Fusion]
    I --> J

    J --> K[Classification Head]
    K --> L[Output Severity Hayashi<br/>Mild / Moderate / Severe / Very Severe]