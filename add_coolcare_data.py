"""
Add Cool & Care company information to Pinecone with proper sections and metadata
"""
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

INDEX_NAME = "cc"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Company information data organized by sections
COMPANY_DATA = {
    "company_overview": {
        "text": """Cool & Care SAS (also known as C&C or Cool & Care SARL) is a comprehensive IT, physical security, fire safety, and HVAC solutions provider operating exclusively in the Democratic Republic of the Congo (DRC). Founded in 2000, the company has established itself as a leading provider of tailored business solutions for enterprises of all sizes—from small businesses to large corporations—operating in one of Africa's most challenging markets.

The company is structured as a Partnership with 11-50 employees and operates under the tagline: "Leading IT Security, Fire Safety and HVAC Solutions Provider in D R Congo". Cool & Care positions itself as "a catalyst for the future of technologies and support for businesses in the DRC" with vast wealth of knowledge and uniquely skilled personnel backed by necessary training.""",
        "metadata": {
            "category": "about",
            "topic": "company overview",
            "keywords": "cool care, company, overview, DRC, founded 2000, IT security, fire safety, HVAC",
            "intent": "learn",
            "section": "company_overview",
            "priority": 10
        }
    },
    "contact_information": {
        "text": """Cool & Care maintains a network of 4+ service locations across the DRC:

Primary Locations:
- Kinshasa (Headquarters): Immeuble 1113, 3ème étage (Unit 2B), Boulevard du 30 Juin, Kinshasa, RD Congo
- Lubumbashi: 7721, Av. Kisambi, Lido Golf, Lubumbashi
- Kolwezi: Q ၃/ Jolle ite, c/ Manika, Route Likasi, Kolwezi
- Additional presence in Goma, Mbuji-Mayi, Matadi, and Likasi

Contact Details:
- Phone: +243 841 364 201
- Email: info@coolcare.cd
- Website: https://www.coolcare.cd or https://coolcare.cd
- Operating Hours: Monday-Saturday, 9:00 AM - 6:00 PM (local time)
- Social Media: Active on Facebook (9.2K likes, 9.3K followers), Instagram (@coolcaredrc), LinkedIn (421 followers)""",
        "metadata": {
            "category": "support",
            "topic": "contact information",
            "keywords": "contact, phone, email, address, location, office, headquarters, lubumbashi, kolwezi, kinshasa",
            "intent": "support",
            "section": "contact",
            "priority": 9
        }
    },
    "it_security_solutions": {
        "text": """IT and Security Solutions:

CCTV Surveillance Systems - The company specializes in state-of-the-art security measures designed for the unique security challenges of organizations operating in the DRC. They believe optimal physical security strategies combine technology and specialized hardware. Key Offerings: General and specialized enterprise surveillance solutions, Video monitoring systems, Security camera installations (500+ cameras installed to date).

Structured Cabling and Networking - Enterprise network communications backbone solutions connecting computers and devices across departments and workgroup networks, facilitating data accessibility. Services include: Network infrastructure design and implementation, Campus and building-wide networking, Global connectivity solutions for branch offices and data centers, Cloud-based communications integration.

Data Centre Setup and Installation - Comprehensive full-life cycle services including: Planning and designing, Supply and installation, Implementation and ongoing maintenance, Complete data center infrastructure solutions.

Access Control + Employee Management - Beyond granting entrance to authorized personnel, these systems provide: Effective employee management solutions, Time and attendance tracking, Integration with existing security infrastructure, Biometric and card-based access systems.

Entrance Management - Field-proven equipment with reputation for reliability and durability, specializing in perimeter access control.

Intrusion Alarms - Easy-to-use systems that seamlessly integrate with existing security infrastructure including: Video cameras, Smoke alarms, Home automation devices, Complete security ecosystem integration.""",
        "metadata": {
            "category": "products",
            "topic": "IT security solutions",
            "keywords": "IT security, CCTV, surveillance, cameras, networking, cabling, data center, access control, intrusion alarms, security systems",
            "intent": "buy",
            "section": "services",
            "priority": 9
        }
    },
    "fire_safety_solutions": {
        "text": """Fire Safety Solutions:

Cool & Care provides firefighting systems and technologies unique to the market, offering complete solutions for every industry. Unlike competitors who offer one-size-fits-all solutions, C&C provides solutions for all types of fires, increasing safety for workspaces, equipment, and personnel.

Fire Detection and Alarm Systems - Advanced detection technology, Early warning systems, Industry-specific fire safety solutions, Integration with building management systems.

Fire Suppression Systems - Comprehensive fire extinguishing solutions, Automatic suppression systems, Manual firefighting equipment, Specialized solutions for different fire classes.""",
        "metadata": {
            "category": "products",
            "topic": "fire safety solutions",
            "keywords": "fire safety, fire detection, fire alarms, fire suppression, fire extinguishers, fire protection, safety systems",
            "intent": "buy",
            "section": "services",
            "priority": 9
        }
    },
    "hvac_solutions": {
        "text": """HVAC Solutions:

Commercial VRF (Variable Refrigerant Flow) Systems - Cool & Care offers a broad range of energy-efficient heating, ventilation, and air conditioning systems, including: Variable Refrigerant Flow systems, Dehumidifying products, Air cleaning systems, Service and parts support, Advanced building controls, Duct building and installation services.

The company works with major HVAC brands including Daikin, Mitsubishi Electric, LG, Hitachi, and Fujitsu—all recognized global leaders in VRF technology.""",
        "metadata": {
            "category": "products",
            "topic": "HVAC solutions",
            "keywords": "HVAC, air conditioning, heating, ventilation, VRF, Daikin, Mitsubishi, LG, Hitachi, Fujitsu, climate control",
            "intent": "buy",
            "section": "services",
            "priority": 9
        }
    },
    "maintenance_service": {
        "text": """Maintenance and After-Sales Service:

To maintain high security and operational efficiency, Cool & Care provides comprehensive maintenance and after-sales service for all supplied equipment during both warranty and post-warranty periods.

Service Coverage: Hardware installations, Software installations, Preventative maintenance programs, Technical guidelines for equipment operation, Emergency repair services, Ongoing technical support.

All customers receive assurance that necessary technical guidelines for equipment operation and maintenance are handled by Cool & Care's trained personnel.""",
        "metadata": {
            "category": "support",
            "topic": "maintenance service",
            "keywords": "maintenance, after-sales, service, support, warranty, repair, technical support, installation",
            "intent": "support",
            "section": "services",
            "priority": 8
        }
    },
    "solar_solutions": {
        "text": """Solar Solutions:

Cool & Care delivers complete solar and backup power solutions for homes, businesses, and unique applications, with capability to customize systems to fit exact needs.

Solar Power Offerings: Solar panel installations, Complete solar power kits, Backup power systems, Energy storage solutions, Custom-designed renewable energy systems, Integration with existing infrastructure.

This service is particularly relevant given the DRC's energy challenges and growing adoption of on-site solar projects at mining operations and other facilities throughout the country.""",
        "metadata": {
            "category": "products",
            "topic": "solar solutions",
            "keywords": "solar, solar panels, solar power, backup power, energy storage, renewable energy, power solutions",
            "intent": "buy",
            "section": "services",
            "priority": 8
        }
    },
    "partnerships": {
        "text": """Strategic Partnerships:

Cool & Care's competitive advantage stems from partnerships with renowned, industry-leading IT and safety brands:

Fire Safety Partners:
- Ceasefire Industries (India): Leading manufacturer of fire extinguishers, in-panel suppression systems, kitchen fire suppression, total flooding solutions, alarm systems, and hydrants. Ceasefire exports to multiple African countries including Congo.
- Dafo Vehicle Fire Protection (Sweden): Since 2020, C&C has been the exclusive partner for Dafo Vehicle's fire suppression systems in the DRC. The partnership brings vehicle fire suppression solutions particularly for mining, forestry, and heavy equipment. Dafo Vehicle Systems Include: BusLine system for commercial vehicles, Vulcan system for challenging demands, safEV solution for electric and hybrid vehicles, Forrex suppression agent (non-corrosive, environmentally friendly, effective cooling).

HVAC Brands: Daikin (Japanese pioneer in VRF technology), Mitsubishi Electric (global leader in advanced HVAC), LG (comprehensive VRF systems), Hitachi (reliable and silent HVAC solutions), Fujitsu (Airstage VRF systems).""",
        "metadata": {
            "category": "about",
            "topic": "partnerships",
            "keywords": "partnerships, partners, Ceasefire, Dafo Vehicle, Daikin, Mitsubishi, LG, Hitachi, Fujitsu, brands",
            "intent": "learn",
            "section": "partnerships",
            "priority": 7
        }
    },
    "target_industries": {
        "text": """Target Industries & Clients:

Cool & Care serves businesses across multiple sectors in the DRC:

Primary Target Markets: Mining Industry (The DRC's dominant sector with copper, cobalt, gold, diamond, lithium, and other mineral operations. Mining companies require robust fire suppression for vehicles and equipment, security systems for remote sites, and reliable HVAC for harsh environments), Enterprise Corporations (Large multinational and national businesses), Medium and Small Entities (Growing businesses requiring scalable solutions), Warehouses (50+ warehouses secured), Industrial Facilities (Manufacturing and processing plants), Commercial Buildings (Offices, retail centers), Government and Institutional Buildings.

Notable Achievement Metrics: 500+ cameras installed, 50+ warehouses secured, 150+ alarms programmed, Experience with World Bank HQ projects.""",
        "metadata": {
            "category": "about",
            "topic": "target industries",
            "keywords": "industries, clients, mining, enterprise, warehouses, industrial, commercial, government, target markets",
            "intent": "learn",
            "section": "industries",
            "priority": 8
        }
    },
    "competitive_advantages": {
        "text": """Competitive Advantages:

1. Local Expertise in Challenging Environment - Cool & Care is intimately familiar with the unique security, infrastructure, and operational challenges in the DRC—including unstable power supply, remote locations, security threats, and harsh environmental conditions.

2. Comprehensive Service Portfolio - Unlike competitors who may specialize in one area, C&C offers integrated solutions spanning IT, security, fire safety, HVAC, and solar power—allowing clients to work with a single trusted provider.

3. Multi-Location Service Network - With offices in Kinshasa, Lubumbashi, Kolwezi, and presence in additional cities, C&C can provide faster response times and better support across the DRC's vast geography.

4. Quality International Partnerships - Partnerships with globally recognized brands like Ceasefire and Dafo Vehicle provide access to world-class technology and proven solutions.

5. Full-Lifecycle Support - From initial consultation and design through installation, commissioning, warranty service, and post-warranty maintenance, C&C handles every phase.

6. Trained and Skilled Personnel - Wide range of uniquely skilled personnel backed by comprehensive training to offer definitive support services.

7. Customized Solutions - Rather than off-the-shelf approaches, C&C provides tailored solutions designed for specific client needs and DRC operating conditions.""",
        "metadata": {
            "category": "about",
            "topic": "competitive advantages",
            "keywords": "advantages, benefits, expertise, local knowledge, comprehensive services, partnerships, support, customized solutions",
            "intent": "learn",
            "section": "advantages",
            "priority": 8
        }
    },
    "service_delivery": {
        "text": """Service Delivery Model:

Consultation Process - Clients can request free consultations through: Website contact form (selecting product category: Fire Safety, IT and Security, HVAC Solutions, Maintenance, Solar Solution), Direct phone contact: +243 841 364 201, Email: info@coolcare.cd, Walk-in visits to any office location.

Project Lifecycle: Initial Consultation (Understanding client needs and challenges), Site Assessment (Evaluating facility requirements), Solution Design (Creating customized system plans), Proposal and Quote (Detailed pricing and specifications), Supply (Procurement of equipment and materials), Installation (Professional implementation by trained technicians), Commissioning (Testing and activation), Training (Operator and maintenance staff training), Warranty Support (Comprehensive warranty period service), Post-Warranty Maintenance (Ongoing service contracts).""",
        "metadata": {
            "category": "support",
            "topic": "service delivery",
            "keywords": "consultation, service delivery, project lifecycle, installation, commissioning, training, warranty, maintenance",
            "intent": "buy",
            "section": "process",
            "priority": 8
        }
    },
    "company_values": {
        "text": """Company Values & Mission:

Quality Customer Service Focus - Quality customer service is at the heart of Cool & Care, driving their creation of a network of sales and service centers to deliver efficient, reliable, and fast service.

Core Commitments: Providing the most reliable and very best fire detection and suppression, Ensuring organizations and personnel are always safe and secure, Offering definitive support services through skilled personnel, Being a catalyst for the future of technologies in the DRC, Maintaining perfect operational levels through expert maintenance.

Company Philosophy: "The best and most viable physical security strategies make use of both technology and specialized hardware to achieve safety goals".""",
        "metadata": {
            "category": "about",
            "topic": "company values",
            "keywords": "values, mission, philosophy, customer service, commitments, quality, safety, security",
            "intent": "learn",
            "section": "values",
            "priority": 7
        }
    },
    "technical_capabilities": {
        "text": """Technical Capabilities:

Fire Safety Expertise: Multi-zone fire protection solutions, Vehicle fire suppression systems, Automatic fire detection and suppression, Manual firefighting equipment, Fire alarm and notification systems, Solutions for different fire classes (electrical, flammable liquids, combustibles, metals).

IT Infrastructure: Enterprise networking (LAN/WAN), Structured cabling systems, Data center design and implementation, Server and storage solutions, Network security infrastructure, Cloud connectivity solutions.

Security Systems: IP and analog CCTV systems, Access control (biometric, card-based, PIN), Intrusion detection and alarm systems, Perimeter security solutions, Integrated security management platforms, Remote monitoring capabilities.

HVAC Expertise: Variable Refrigerant Flow (VRF) systems, Commercial air conditioning, Industrial ventilation, Climate control solutions, Energy-efficient systems, Building automation integration.

Solar & Power: Photovoltaic system design, Battery storage solutions, Hybrid power systems, Grid-tied and off-grid configurations, Backup power solutions, Power quality management.""",
        "metadata": {
            "category": "products",
            "topic": "technical capabilities",
            "keywords": "technical, capabilities, expertise, fire safety, IT infrastructure, security systems, HVAC, solar, power",
            "intent": "learn",
            "section": "technical",
            "priority": 8
        }
    }
}

def add_coolcare_data(namespace: str = None):
    """
    Add Cool & Care company data to Pinecone with proper sections and metadata
    
    Args:
        namespace: Optional namespace (if None, uses default)
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Create vectorstore
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    
    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    all_chunks = []
    all_metadata = []
    
    print("Processing Cool & Care company data...")
    print(f"Total sections: {len(COMPANY_DATA)}\n")
    
    for section_name, section_data in COMPANY_DATA.items():
        text = section_data["text"]
        base_metadata = section_data["metadata"]
        
        # Split text into chunks
        chunks = splitter.split_text(text)
        
        print(f"Section: {section_name}")
        print(f"  - Text length: {len(text)} chars")
        print(f"  - Chunks created: {len(chunks)}")
        print(f"  - Category: {base_metadata['category']}")
        print(f"  - Topic: {base_metadata['topic']}")
        print(f"  - Intent: {base_metadata['intent']}")
        print()
        
        # Create metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata['url'] = 'https://www.coolcare.cd'
            chunk_metadata['page_title'] = f"Cool & Care - {base_metadata['topic'].title()}"
            chunk_metadata['type'] = 'text'
            chunk_metadata['chunk_index'] = i
            chunk_metadata['section'] = section_name
            
            all_chunks.append(chunk)
            all_metadata.append(chunk_metadata)
    
    # Add to vectorstore in batches
    batch_size = 100
    total_added = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        batch_meta = all_metadata[i:i+batch_size]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)
        total_added += len(batch_texts)
        print(f"✅ Added batch {i//batch_size + 1} ({len(batch_texts)} chunks)")
    
    print(f"\n🎉 Successfully added {total_added} chunks from {len(COMPANY_DATA)} sections!")
    print(f"Namespace: '{namespace or 'default'}'")
    print("\nSections added:")
    for section_name in COMPANY_DATA.keys():
        print(f"  - {section_name}")
    
    return total_added

if __name__ == "__main__":
    print("="*60)
    print("Cool & Care Data Ingestion")
    print("="*60)
    print()
    
    # Add to default namespace (or specify a namespace)
    add_coolcare_data(namespace=None)
    
    print("\n✅ Data ingestion complete!")
    print("\nThe data is now searchable with proper metadata:")
    print("  - Category-based filtering")
    print("  - Topic-based search")
    print("  - Intent detection")
    print("  - Section-based retrieval")

