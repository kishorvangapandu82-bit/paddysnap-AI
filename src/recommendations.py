"""
============================================================
PADDY GUARD AI -- MODULE 20 & 21: Agronomy Knowledge Base
============================================================
Purpose:
    Provides scientifically grounded disease descriptions, visual
    symptoms, chemical/organic treatment guidelines, and
    fertilizer/nutrient recommendations for all 10 paddy classes.
============================================================
"""

from typing import Dict, Any

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "bacterial_leaf_blight": {
        "common_name": "Bacterial Leaf Blight (BLB)",
        "causal_organism": "Xanthomonas oryzae pv. oryzae (Bacteria)",
        "severity": "High (can cause 20-50% yield loss)",
        "symptoms": [
            "Water-soaked lesions on leaf tips that enlarge and turn yellowish-white with wavy margins.",
            "Leaves turn straw-yellow, dry up, and curl longitudinally (Kresek stage in young plants).",
            "Bacterial ooze droplets visible on infected leaf surfaces under humid morning conditions."
        ],
        "causes": [
            "High temperature (25-34°C) combined with high relative humidity (>80%).",
            "Excessive application of nitrogenous fertilizers.",
            "Standing water in fields and strong winds/rainstorms causing leaf micro-wounds."
        ],
        "organic_cultural_management": [
            "Use certified disease-resistant paddy varieties (e.g., IR64, Improved Samba Mahsuri).",
            "Ensure field drainage; avoid continuous flooding during active disease outbreak.",
            "Apply neem cake @ 250 kg/ha and spray Pseudomonas fluorescens (0.2%) @ 10-day intervals.",
            "Destroy and burn infected crop stubble and alternate host weeds."
        ],
        "chemical_control": [
            "Copper hydroxide 77% WP @ 2 g/L or Copper oxychloride 50% WP @ 2.5 g/L.",
            "Streptocycline (90:10 Streptomycin + Tetracycline) @ 0.1 g/L combined with copper spray.",
            "Avoid spraying during flowering/pollination hours to protect beneficial pollinators."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "IMMEDIATELY REDUCE / SPLIT nitrogen application. Excess N creates soft, lush vegetative tissues highly vulnerable to bacterial entry.",
            "potassium_action": "INCREASE Potassium (MOP @ 50-60 kg/ha). Potassium strengthens epidermal cell walls and silica layers, boosting disease resistance.",
            "phosphorus_action": "Apply balanced basal phosphorus based on soil test.",
            "micronutrients": "Foliar spray of Zinc Sulfate (0.5%) + Ferrous Sulfate (0.5%) to restore chlorophyll synthesis."
        }
    },

    "bacterial_leaf_streak": {
        "common_name": "Bacterial Leaf Streak (BLS)",
        "causal_organism": "Xanthomonas oryzae pv. oryzicola (Bacteria)",
        "severity": "Moderate to High",
        "symptoms": [
            "Narrow, translucent interveinal streaks that turn yellow to brownish-orange.",
            "Minute yellow beads of bacterial exudate on lesions.",
            "Entire leaf blades may turn brown and die in severe infections."
        ],
        "causes": [
            "High humidity, frequent rainfall, temperatures 28-32°C.",
            "Excessive nitrogen application and mechanical injury to leaf blades."
        ],
        "organic_cultural_management": [
            "Practice wider plant spacing (20 × 15 cm) to improve field aeration.",
            "Seed treatment with hot water (52-54°C for 15 minutes) or bio-agents."
        ],
        "chemical_control": [
            "Spray Copper oxychloride @ 2.5 g/L + Streptocycline @ 0.1 g/L.",
            "Ensure thorough canopy coverage."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Apply nitrogen in 3-4 split doses rather than heavy single basal application.",
            "potassium_action": "Apply supplementary Potassium (K2O) to thicken leaf cuticle layers.",
            "phosphorus_action": "Standard balanced application.",
            "micronutrients": "Foliar application of Silicon / Potassium Silicate to increase leaf rigidity."
        }
    },

    "bacterial_panicle_blight": {
        "common_name": "Bacterial Panicle Blight (BPB)",
        "causal_organism": "Burkholderia glumae / Burkholderia gladioli (Bacteria)",
        "severity": "High (direct grain loss up to 75%)",
        "symptoms": [
            "Discoloration and blighting of panicles and florets during heading.",
            "Florets turn white/tan with reddish-brown borders; grains remain empty or aborted.",
            "Panicles remain upright instead of bending down due to unfilled grains."
        ],
        "causes": [
            "Extremely hot night temperatures (>28°C) during flowering/heading stage.",
            "Frequent rainfall and high canopy humidity at anthesis."
        ],
        "organic_cultural_management": [
            "Adjust transplanting dates to avoid flowering during peak summer heat waves.",
            "Use pathogen-free seed stocks and resistant cultivars."
        ],
        "chemical_control": [
            "Oxolinic acid seed treatment and heading-stage bactericide applications where registered.",
            "Copper-based bactericide sprays at boot-leaf and early heading stages."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Strictly avoid late-season nitrogen top-dressing after panicle initiation.",
            "potassium_action": "Apply Potassium before panicle initiation to enhance vascular transport into grains.",
            "phosphorus_action": "Ensure adequate phosphorus during early root development.",
            "micronutrients": "Boron foliar spray (0.1%) during pre-flowering to improve pollen viability and grain set."
        }
    },

    "blast": {
        "common_name": "Rice Blast (Leaf / Neck / Node Blast)",
        "causal_organism": "Magnaporthe oryzae / Pyricularia oryzae (Fungus)",
        "severity": "Severe (most destructive fungal disease worldwide)",
        "symptoms": [
            "Spindle-shaped or diamond-shaped lesions with grayish-white centers and dark brown margins.",
            "Lesions coalesce, causing complete withering and death of entire leaves (Leaf Blast).",
            "Neck rot causing panicle breakage and completely blank grains (Neck Blast)."
        ],
        "causes": [
            "Low night temperatures (18-24°C) with high relative humidity (>90%) and prolonged leaf wetness.",
            "Excessive nitrogen application and dense canopy spacing."
        ],
        "organic_cultural_management": [
            "Plant blast-resistant paddy varieties.",
            "Avoid nursery flooding and excessive seedling density.",
            "Seed treatment with Trichoderma harzianum @ 10 g/kg seed."
        ],
        "chemical_control": [
            "Tricyclazole 75% WP @ 0.6 g/L (highly effective preventative and curative fungicide).",
            "Isoprothiolane 40% EC @ 1.5 mL/L or Azoxystrobin 23% SC @ 1.0 mL/L.",
            "Spray at first appearance of leaf spots and repeat at boot-leaf stage."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "SUSPEND nitrogen top-dressing immediately. High nitrogen directly increases leaf silica deficiency and blast susceptibility.",
            "potassium_action": "Apply Muriate of Potash (MOP) @ 50 kg/ha in splits. High K enhances physiological blast resistance.",
            "phosphorus_action": "Maintain optimal phosphorus; avoid severe deficiency.",
            "micronutrients": "Silica foliar spray (soluble potassium silicate @ 2.5 mL/L) to create physical epidermal barrier against fungal appressoria penetration."
        }
    },

    "brown_spot": {
        "common_name": "Brown Spot (Sesame Leaf Spot)",
        "causal_organism": "Bipolaris oryzae / Helminthosporium oryzae (Fungus)",
        "severity": "Moderate to High (Indicator of Soil Nutrient Depletion)",
        "symptoms": [
            "Small, oval to circular dark brown spots resembling sesame seeds across leaf blades.",
            "Larger spots develop gray centers with prominent yellow chlorotic halos.",
            "Infected glumes and grains develop dark brown spots and become chalky."
        ],
        "causes": [
            "Nutrient-deficient, unfertilized, or poorly drained soils (historically linked to Bengal Famine).",
            "Drought stress followed by high humidity and temperatures 25-30°C."
        ],
        "organic_cultural_management": [
            "Soil testing to identify and rectify underlying macro and micronutrient deficiencies.",
            "Apply farmyard manure (FYM) or green manure @ 10-12 tonnes/ha to enrich soil organic matter.",
            "Seed soaking in hot water (53°C for 10 min) or Trichoderma viride @ 5 g/kg."
        ],
        "chemical_control": [
            "Mancozeb 75% WP @ 2.5 g/L or Propiconazole 25% EC @ 1.0 mL/L.",
            "Carbendazim 50% WP @ 1.0 g/L sprayed at tillering and panicle emergence."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Apply balanced nitrogen in 3 split doses. Brown spot often strikes nutrient-starved, nitrogen-deficient crops.",
            "potassium_action": "MANDATORY Potassium application (MOP @ 50 kg/ha). Potassium deficiency is the single largest predisposing factor for brown spot.",
            "phosphorus_action": "Apply recommended Single Super Phosphate (SSP) at basal stage.",
            "micronutrients": "Zinc Sulfate @ 25 kg/ha basal + Silicon foliar spray to strengthen leaf structure."
        }
    },

    "dead_heart": {
        "common_name": "Dead Heart / White Ear (Stem Borer Infestation)",
        "causal_organism": "Scirpophaga incertulas (Yellow Stem Borer Insect Larva)",
        "severity": "High (direct tiller and panicle loss)",
        "symptoms": [
            "Central vegetative shoot turns yellow, dries up, and easily pulls out by hand (Dead Heart at vegetative stage).",
            "Panicles emerge completely white, upright, and empty with no grain filling (White Earhead at reproductive stage).",
            "Tiny entry/exit holes with larval frass visible on lower stem nodes."
        ],
        "causes": [
            "Continuous monocropping, staggered planting, and excessive nitrogenous fertilization.",
            "Warm humid weather favoring stem borer moth emergence and egg laying."
        ],
        "organic_cultural_management": [
            "Install Pheromone Traps @ 8-10 traps/ha for monitoring and mass trapping male moths.",
            "Release egg parasitoids (Trichogramma japonicum) @ 100,000 wasps/ha weekly for 4 weeks.",
            "Clip seedling leaf tips before transplanting to eliminate egg masses.",
            "Encourage natural predators (dragonflies, spiders, frogs)."
        ],
        "chemical_control": [
            "Chlorantraniliprole 0.4% G @ 10 kg/ha (broadcast) or Chlorantraniliprole 18.5% SC @ 0.3 mL/L foliar.",
            "Cartap Hydrochloride 50% SP @ 2 g/L or Fipronil 5% SC @ 2 mL/L.",
            "Apply when egg masses exceed economic threshold level (ETL: 1 egg mass/m² or 5% dead hearts)."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Avoid excessive nitrogen application which produces succulent, tender stems highly attractive to boring larvae.",
            "potassium_action": "Ensure adequate potassium to promote thick, lignified stem culms that resist larval boring.",
            "phosphorus_action": "Standard balanced application.",
            "micronutrients": "Silicon fertilizer (calcium silicate) applied to soil to increase stem silica content."
        }
    },

    "downy_mildew": {
        "common_name": "Downy Mildew / Yellow Dwarf / Crazy Rice",
        "causal_organism": "Sclerophthora macrospora (Oomycete)",
        "severity": "Moderate (localized in waterlogged areas)",
        "symptoms": [
            "Stunted plants with severe twisting, curling, and crinkling of leaves.",
            "White to yellowish chlorotic specks and streaks on leaves.",
            "Deformed, leafy panicles with distorted florets and no viable grains."
        ],
        "causes": [
            "Submerged or poorly drained lowland nurseries and fields.",
            "Cool wet weather with prolonged waterlogging during early seedling stages."
        ],
        "organic_cultural_management": [
            "Improve nursery drainage; avoid submerged seedbeds.",
            "Rogue out and destroy infected stunted plants early."
        ],
        "chemical_control": [
            "Metalaxyl-M + Mancozeb @ 2.5 g/L as foliar spray.",
            "Fosetyl-Al @ 2.0 g/L for systemic oomycete suppression."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Balanced nitrogen; avoid delayed applications in waterlogged patches.",
            "potassium_action": "Adequate potassium to support vascular health under wet soil conditions.",
            "phosphorus_action": "Ensure adequate phosphorus for strong seedling root vigor.",
            "micronutrients": "Foliar micronutrient spray (Fe + Mn) to correct chlorosis in wet soils."
        }
    },

    "hispa": {
        "common_name": "Rice Hispa Infestation",
        "causal_organism": "Dicladispa armigera (Spiny Beetle Insect)",
        "severity": "Moderate to High",
        "symptoms": [
            "Parallel white streaks on the upper leaf surface caused by adult beetles scraping epidermal green tissues.",
            "Blister-like leaf mines caused by larvae feeding internally inside leaf tissues.",
            "Damaged leaves look whitish, membranous, dry up, and give a scorched/burnt appearance to the field."
        ],
        "causes": [
            "Warm humid conditions, heavy rainfall followed by sunny spells.",
            "Excessive nitrogen application and dense canopy without weed control."
        ],
        "organic_cultural_management": [
            "Clip and destroy top 3-4 inches of seedling leaves before transplanting to remove hispa eggs.",
            "Use sweep nets in early mornings to catch and destroy adult beetles.",
            "Maintain clean bunds and remove alternate weed hosts (Echinochloa grass)."
        ],
        "chemical_control": [
            "Chlorpyrifos 20% EC @ 2.5 mL/L or Quinalphos 25% EC @ 2.0 mL/L.",
            "Thiamethoxam 25% WG @ 0.2 g/L for systemic leaf protection.",
            "Apply when hispa damage exceeds ETL (1 adult or 1 damaged leaf per hill)."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Split nitrogen application into smaller doses; avoid lush green vegetative surges that attract hispa swarms.",
            "potassium_action": "Potassium supplementation to harden leaf epidermis and reduce scraping damage.",
            "phosphorus_action": "Standard balanced application.",
            "micronutrients": "Maintain balanced micronutrient fertility."
        }
    },

    "normal": {
        "common_name": "Healthy Paddy Leaf (No Disease Detected)",
        "causal_organism": "None (Healthy Crop)",
        "severity": "Optimal Crop Health",
        "symptoms": [
            "Uniform green leaf pigmentation with smooth, undamaged blade margins.",
            "No chlorosis, necrotic lesions, leaf mines, or insect scraping marks.",
            "Normal erect canopy structure and healthy tillering."
        ],
        "causes": [
            "Balanced agricultural practices, proper nutrition, and effective pest monitoring."
        ],
        "organic_cultural_management": [
            "Continue standard Integrated Pest Management (IPM) practices.",
            "Maintain optimal water depth (2-5 cm during vegetative and reproductive stages).",
            "Periodic monitoring of field bunds and light traps for early pest detection."
        ],
        "chemical_control": [
            "NO CHEMICAL SPRAY REQUIRED.",
            "Avoid unnecessary prophylactic fungicide/pesticide sprays to protect beneficial field fauna."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Follow standard N-P-K recommendation (e.g., 120:60:60 kg/ha for high-yielding varieties). Apply N in 3 equal splits (Basal, Active Tillering, Panicle Initiation).",
            "potassium_action": "Apply 50% K as basal and 50% at panicle initiation for optimal grain filling and weight.",
            "phosphorus_action": "Apply 100% Phosphorus as basal at the time of final puddle preparation.",
            "micronutrients": "Apply Zinc Sulfate @ 25 kg/ha once every two seasons to maintain high tillering capacity."
        }
    },

    "tungro": {
        "common_name": "Rice Tungro Disease (RTD)",
        "causal_organism": "Rice Tungro Bacilliform Virus (RTBV) + Spherical Virus (RTSV) transmitted by Green Leafhopper (Nephotettix virescens)",
        "severity": "Severe (can cause total crop failure)",
        "symptoms": [
            "Yellowing and orange discoloration starting from leaf tips and extending downwards along margins.",
            "Severe plant stunting, reduced tillering, delayed flowering, and incomplete panicle emergence.",
            "Grains are sterile, discolored, or poorly filled."
        ],
        "causes": [
            "High population of vector Green Leafhoppers (GLH).",
            "Staggered planting enabling continuous virus transmission across adjacent fields."
        ],
        "organic_cultural_management": [
            "Synchronized planting in the community to break the vector life cycle.",
            "Use tungro-resistant varieties (e.g., Vikramarya, IR36).",
            "Install yellow sticky traps and light traps to monitor and trap Green Leafhoppers.",
            "Rogue out and destroy infected virus-source hills immediately."
        ],
        "chemical_control": [
            "Control GLH vector: Imidacloprid 17.8% SL @ 0.3 mL/L or Dinotefuran 20% SG @ 0.4 g/L.",
            "Thiamethoxam 25% WG @ 0.2 g/L sprayed at nursery and tillering stages."
        ],
        "fertilizer_nutrient_guidance": {
            "nitrogen_action": "Do NOT over-apply nitrogen. Yellowing can be mistaken for N-deficiency; adding N will attract more leafhopper vectors.",
            "potassium_action": "Foliar spray of Potassium Nitrate (1%) + Micronutrients to support recovering plants after vector control.",
            "phosphorus_action": "Standard balanced application.",
            "micronutrients": "Foliar spray of 2% Urea + 1% Zinc Sulfate to stimulate recovery in mildly affected fields after vector eradication."
        }
    }
}


def get_recommendation(disease_key: str) -> Dict[str, Any]:
    """Returns complete agronomy and management details for a given disease class."""
    clean_key = disease_key.lower().strip().replace(" ", "_").replace("-", "_")
    return DISEASE_KNOWLEDGE_BASE.get(clean_key, DISEASE_KNOWLEDGE_BASE["normal"])
