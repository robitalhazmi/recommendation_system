# Predictive Analytics Web Application

## Overview

This web application uses predictive analytics to recommend products for a given customer based on historical data. It is designed for an e-commerce company to enhance its marketing strategy. The demo can be accessed in https://robit.pythonanywhere.com.

## Features

- Predictive analytics powered by neural collaborative filtering.
- User-friendly web interface for entering customer ID and receiving product recommendations.
- Visual representation of recommended products with images.

## Prerequisites

- Python (3.10 or later)
- Flask
- Flask-Cors
- Numpy
- Requests
- Tensorflow
- Keras
- Pandas (for the example neural collaborative filtering implementation)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

## Data Description
The dataset includes the following information:
1. **Customer details:**
   - user_id: Unique numerical identifier assigned to each user.
   - user_name: User's name or username.

   Data sample:
   | user_id                      | user_name     |
   |------------------------------|---------------|
   | AFC3FFC5PKFF5PMA52S3VCHOZ5FQ | S.ARUMUGAM    |
   | AECPFYFQVRUWC3KGNLJIOREFP5LQ | ArdKn         |
   | AE3MQNNHHLUHXURL5S7IAR7JTGNQ | Durai Vignesh |
   | AFV7ZA733ZLME4KNLZPMPCBUNPPA | Aman          |
   | AEHQYGI5L4FFALBMC5XMT5KXSZCA | Vivek         |

2. **Product details:**
   - product_id: Unique numerical identifier assigned to each product.
   - product_name: The name or title of the product.
   - category_1: The primary category to which the product belongs.
   - category_2: The secondary category to which the product belongs.
   - about_product: A brief description or information about the product.
   - img_link: The URL link to an image representing the product.
   - product_link: The URL link to access detailed information or purchase the product.
   - discounted_price: The price of the product after applying any discounts.
   - actual_price: The original or undiscounted price of the product.
   - difference_price: The difference between the actual and discounted prices.
   - discount_percentage: The percentage discount applied to the product.

   Data sample:
   | product_id | product_name | category_1 | category_2 | about_product | img_link | product_link | discounted_price | actual_price | difference_price | discount_percentage |
   |------------|--------------|------------|------------|-------------- |----------|--------------|------------------|--------------|------------------|---------------------|
   | B098NS6PVG | Ambrane Unbreakable 60W / 3A Fast Charging 1.5m Braided Type C Cable for Smartphones, Tablets, Laptops & other Type C devices, PD Technology, 480Mbps Data Sync, Quick Charge 3.0 (RCT15A, Black) | Computers & Accessories | Accessories & Peripherals | Compatible with all Type C enabled devices, be it an android smartphone (Mi, Samsung, Oppo, Vivo, Realme, OnePlus, etc), tablet, laptop (Macbook, Chromebook, etc) Supports Quick Charging (2.0/3.0) Unbreakable – Made of special braided outer with rugged interior bindings, it is ultra-durable cable that won’t be affected by daily rough usage Ideal Length – It has ideal length of 1.5 meters which is neither too short like your typical 1meter cable or too long like a 2meters cable Supports maximum 3A fast charging and 480 Mbps data transfer speed 6 months manufacturer warranty from the date of purchase | [img1](https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/31zOsqQOAOL._SY445_SX342_QL70_FMwebp_.jpg) | [prod1](https://www.amazon.in/Ambrane-Unbreakable-Charging-Braided-Cable/dp/B098NS6PVG/ref=sr_1_2?qid=1672909124&s=electronics&sr=1-2) | 199.0 | 349.0 | 150.0 | 0.43 |
   | B096MSW6CT | Sounce Fast Phone Charging Cable & Data Sync USB Cable Compatible for iPhone 13, 12,11, X, 8, 7, 6, 5, iPad Air, Pro, Mini & iOS Devices | Computers & Accessories | Accessories & Peripherals | 【 Fast Charger& Data Sync】-With built-in safety proctections and four-core copper wires promote maximum signal quality and strength and enhance charging & data transfer speed with up to 480 mb/s transferring speed.【 Compatibility】-Compatible with iPhone 13, 12,11, X, 8, 7, 6, 5, iPad Air, Pro, Mini & iOS devices.【 Sturdy & Durable】-The jacket and enforced connector made of TPE and premium copper, are resistant to repeatedly bending and coiling.【 Ultra High Quality】: According to the experimental results, the fishbone design can accept at least 20,000 bending and insertion tests for extra protection and durability. Upgraded 3D aluminum connector and exclusive laser welding technology, which to ensure the metal part won't break and also have a tighter connection which fits well even with a protective case on and will never loose connection.【 Good After Sales Service】-Our friendly and reliable customer service will respond to you within 24 hours ! you can purchase with confidence,and every sale includes a 365-day worry-free Service to prove the importance we set on quality. | [img2](https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/31IvNJZnmdL._SY445_SX342_QL70_FMwebp_.jpg) | [prod2](https://www.amazon.in/Sounce-iPhone-Charging-Compatible-Devices/dp/B096MSW6CT/ref=sr_1_3?qid=1672909124&s=electronics&sr=1-3) | 199.0 | 1899.0 | 1700.0 | 0.90 |
   | B08HDJ86NZ | boAt Deuce USB 300 2 in 1 Type-C & Micro USB Stress Resistant, Tangle-Free, Sturdy Cable with 3A Fast Charging & 480mbps Data Transmission, 10000+ Bends Lifespan and Extended 1.5m Length(Martian Red) | Computers & Accessories | Accessories & Peripherals | The boAt Deuce USB 300 2 in 1 cable is compatible with smartphones, tablets, PC peripherals, Bluetooth speakers, power banks and all other devices with Type-C as well as Micro USB port It ensures 3A fast charging and data transmissions with rapid sync at 480 mbps The premium Nylon braided skin makes it sturdy and invincible against external damage Its Aluminium alloy shell housing makes it last longer with 10000+ Bends Lifespan with extended frame protection for strain relief The resilient and flexible design offers a tangle free experience seamlessly Deuce USB 300 cable offers a perfect 1.5 meters in length for smooth & hassle-free user experience 2 years warranty from the date of purchase | [img3](https://m.media-amazon.com/images/I/41V5FtEWPkL._SX300_SY300_QL70_FMwebp_.jpg) | [prod3](https://www.amazon.in/Deuce-300-Resistant-Tangle-Free-Transmission/dp/B08HDJ86NZ/ref=sr_1_4?qid=1672909124&s=electronics&sr=1-4) | 329.0 | 699.0 | 370.0 | 0.53 |
   | B08HDJ86NZ | Portronics Konnect L 1.2M Fast Charging 3A 8 Pin USB Cable with Charge & Sync Function for iPhone, iPad (Grey) | Computers & Accessories | Accessories & Peripherals | [CHARGE & SYNC FUNCTION]- This cable comes with charging & Data sync function [HIGH QUALITY MATERIAL]- TPE + Nylon Material to make sure that the life of the cable is enhanced significantly [LONG CORD]- The Cable is extra thick 1.2 meter long, optimized for an easy use for your comfort at home or office [MORE DURABLE]-This cable is unique interms of design and multi-use and is positioned to provide the best comfort and performance while using [UNIVERSAL COMPATIBILITY]- Compatible with all devices like iPhone XS, X, XR, 8, 7, 6S, 6, 5S, iPad Pro, iPad mini and iPad Air | [img4](https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/31VzNhhqifL._SX300_SY300_QL70_FMwebp_.jpg) | [prod4](https://www.amazon.in/Portronics-Konnect-POR-1080-Charging-Function/dp/B08CF3B7N1/ref=sr_1_5?qid=1672909124&s=electronics&sr=1-5) | 154.0 | 399.0 | 245.0 | 0.61 |
   | B08HDJ86NZ | pTron Solero TB301 3A Type-C Data and Fast Charging Cable, Made in India, 480Mbps Data Sync, Strong and Durable 1.5-Meter Nylon Braided USB Cable for Type-C Devices for Charging Adapter (Black) | Computers & Accessories | Accessories & Peripherals | Fast Charging & Data Sync: Solero TB301 Type-C cable supports fast charge up to 5V/3A for devices and data syncing speed up to 480Mbps. Universal Compatibility: This USB charging cable connects USB Type-C devices with standard USB devices like laptops, hard drives, power banks, wall and car chargers, etc..Connector One: Reversible Type C Connector Two: USB A Type Rough & Tough Type-C Cable: Charging cable with a double-braided exterior, premium aramid fiber core and metal plugs. It has passed 10,000 bending tests and can easily withstand daily use. Extended Length: 1.5-meter long c-type cable uses nylon material to protect the wire and avoid knots. Perfect Fit Connectors: pTron Soler USB-C has passed the 5KG load test, swing test, 5000+ times connect & disconnect to ensure that there are no loose connections. | [img5](https://www.amazon.in/Solero-TB301-Charging-480Mbps-1-5-Meter/dp/B08Y1TFSP6/ref=sr_1_6?qid=1672909124&s=electronics&sr=1-6) | [prod5](https://www.amazon.in/Solero-TB301-Charging-480Mbps-1-5-Meter/dp/B08Y1TFSP6/ref=sr_1_6?qid=1672909124&s=electronics&sr=1-6) | 149.0 | 1000.0 | 851.0 | 0.85 |

3. **Rating history:**
   - user_id: Unique numerical identifier assigned to each user.
   - product_id: Unique numerical identifier assigned to each product.
   - rating: The categorical rating given by the user (e.g., 1 to 5 stars).
   - rating_score: A numerical representation of the user's rating (e.g., 3.5 out of 5).
   - timestamp: The timestamp indicating when the rating was provided.

   Data sample:
   | user_id                      | product_id    | rating | rating_score  | timestamp  |
   |------------------------------|---------------|--------|---------------|------------|
   | AFWTGD4FCS2E2U2TDCOEOGP2FWEA | B097R3XH9R    | 4.0    | Above Average | 964982703  |
   | AGPQHMB6XWAURLOJA57DPCU4HQ7A | B097R3XH9R    | 4.0    | Above Average | 847434962  |
   | AFC4X5UHL2LN4PBS2TWOMIZ2GHAQ | B097R3XH9R    | 4.5    | Above Average | 1106635946 |
   | AFK6D62HRZSHP5W3DE5QGYUYJQEA | B097R3XH9R    | 2.5    | Below Average | 1510577970 |
   | AHQ7LIIQZN6O7YA3EYZ7SV2RIYFQ | B097R3XH9R    | 4.5    | Above Average | 1305696483 |

## Usage

1. **Run the Flask application:**
2. **Open your web browser and go to http://localhost:5000.**
3. **Enter a customer ID in the form and click "Predict."**
4. **View the recommended products with images.**

## Customization

- To customize the predictive model, modify the `neural_collaborative_filtering` function in `app.py`.
- Update the HTML and CSS files in the `templates` and `static` directories for UI customization.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
