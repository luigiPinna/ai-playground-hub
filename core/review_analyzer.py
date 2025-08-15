"""
WORKFLOW SPIEGAZIONE - Booking Review Analyzer

INPUT: Recensioni Booking con contenuto_positivo e contenuto_negativo separati

STEP 1 - SENTIMENT ANALYSIS (LLM):
- Usa modello HuggingFace per analizzare il sentiment di ogni parte
- Risultato: POSITIVE/NEGATIVE/NEUTRAL + confidence score
- Ci dice SE i clienti sono soddisfatti overall

STEP 2 - KEYWORD EXTRACTION (KeyBERT):
- Estrae le parole/frasi più significative da ogni parte
- Positive keywords: cosa apprezzano (staff, piscina, colazione...)
- Negative keywords: cosa criticano (affollata, lento, sporco...)
- Ci dice COSA SPECIFICAMENTE apprezzano/criticano

STEP 3 - CATEGORIZZAZIONE BUSINESS:
- Associa keywords a categorie hotel (staff_service, facilities, room_quality...)
- Conta menzioni per categoria per identificare pattern
- Ci dice DOVE concentrare gli sforzi di miglioramento

OUTPUT FINALE - BUSINESS INTELLIGENCE:
- Summary: sentiment generale + percentuale positive
- Strengths: punti di forza più menzionati (actionable per marketing)
- Areas for improvement: problemi specifici da risolvere (actionable per management)
- Detailed reviews: analisi granulare per ogni recensione

VALORE: Il sentiment da solo dice "82% positive". Con keywords sappiamo che
"i clienti amano lo staff (15 menzioni) ma si lamentano della piscina affollata (5 menzioni)"
→ Azione concreta: gestire meglio accessi piscina in alta stagione

Riassunto:

1. Sentiment Analysis con LLM

Modello: cardiffnlp/twitter-xlm-roberta-base-sentiment
Analizza separatamente contenuto positivo e negativo
Output: POSITIVE/NEGATIVE/NEUTRAL + confidence score

2. Keyword Extraction con KeyBERT

Modello: distilbert-base-multilingual-cased (via KeyBERT)
Estrae keywords semanticamente significative dal testo
Output: liste di keywords/phrases rilevanti

3. Categorizzazione con Regex/Dictionary

Non per verificare aspetti, ma per classificare le keywords
Associa keywords a categorie business: staff_service, facilities, room_quality, etc.
Keywords mapping: {'staff': ['staff', 'personale', 'servizio', ...]}

4. Business Intelligence

Combina sentiment + keywords + categories
Conta menzioni per categoria
Genera insights: "staff eccellente (9 menzioni positive)"
"""
import re
from collections import defaultdict, Counter
from transformers import pipeline
from keybert import KeyBERT
from constants import SAMPLE_REVIEWS


class BookingReviewAnalyzer:
    def __init__(self):
        """
        Professional analyzer for Booking.com reviews.
        Designed for hospitality business intelligence.
        """
        print("Loading sentiment analysis model for hospitality reviews...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )

        print("Loading KeyBERT model for intelligent keyword extraction...")
        self.kw_model = KeyBERT('distilbert-base-multilingual-cased')

        print("Models loaded! Ready to analyze guest feedback 🏨")

        # Categories mapping for hospitality industry
        self.categories = {
            'staff_service': [
                'staff', 'personale', 'servizio', 'accoglienza', 'reception',
                'gentile', 'cortese', 'professionale', 'disponibile', 'danilo'
            ],
            'cleanliness': [
                'pulizia', 'pulito', 'pulitissimo', 'pulite', 'igiene', 'ordinato'
            ],
            'facilities': [
                'piscina', 'spa', 'wellness', 'palestra', 'terrazza', 'ristorante',
                'bar', 'padel', 'idromassaggio', 'aria condizionata'
            ],
            'room_quality': [
                'camera', 'stanza', 'letto', 'balcone', 'vista', 'moderna',
                'elegante', 'ampia', 'insonorizzata', 'condizionatore'
            ],
            'breakfast': [
                'colazione', 'breakfast', 'dolce', 'salata', 'scelta', 'qualità'
            ],
            'location': [
                'posizione', 'villa pamphili', 'parco', 'centro', 'navetta',
                'trasporto', 'fermata', 'collegamento', 'parcheggio'
            ],
            'value_price': [
                'prezzo', 'rapporto', 'qualità/prezzo', 'valore', 'costo'
            ]
        }

    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text"""
        if not text or text.strip() == "":
            return {"sentiment": "Neutral", "confidence": 0.0}

        result = self.sentiment_analyzer(text)

        # Better sentiment mapping for hospitality context
        sentiment_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        raw_label = result[0]['label']
        sentiment = sentiment_map.get(raw_label, 'Positive')  # Default to positive for unknown
        confidence = round(result[0]['score'], 3)

        return {"sentiment": sentiment, "confidence": confidence}

    def _categorize_text(self, text):
        """Identify which categories are mentioned in the text"""
        if not text:
            return []

        text_lower = text.lower()
        mentioned_categories = []

        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    mentioned_categories.append(category)
                    break

        return list(set(mentioned_categories))  # Remove duplicates

    def _extract_keywords(self, text, is_negative=False):
        """
        Extract relevant keywords using KeyBERT NLP model from Hugging Face.
        Much more intelligent than regex-based extraction.
        """
        if not text or len(text.strip()) < 10:
            return []

        try:
            # Use KeyBERT to extract semantically meaningful keywords
            # Simplified version without problematic parameters
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  # Single words and 2-word phrases
                use_maxsum=False,  # Disable MaxSum to simplify
                nr_candidates=10  # Fewer candidates
            )

            # Extract just the keyword strings (not the scores)
            keyword_list = [kw[0] for kw in keywords]

            # Additional filtering for hospitality context
            if is_negative:
                # For negative reviews, focus on actionable issues
                hospitality_issues = [
                    'rumoroso', 'piccolo', 'affollata', 'disorganizzato', 'lento',
                    'sporco', 'freddo', 'caldo', 'rotto', 'mancante', 'difficile'
                ]
                filtered_keywords = []
                for kw in keyword_list:
                    # Keep hospitality-specific issues or any 2-word phrases (more specific)
                    if any(issue in kw.lower() for issue in hospitality_issues) or ' ' in kw:
                        filtered_keywords.append(kw)
                    # Also keep if it's a facility with problem context
                    elif any(facility in kw.lower() for facility in
                             ['piscina', 'colazione', 'camera', 'servizio', 'trasporto']):
                        filtered_keywords.append(kw)

                return filtered_keywords[:3]
            else:
                # For positive reviews, keep quality and facility keywords
                return keyword_list[:5]

        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            # Fallback to simple extraction if KeyBERT fails
            return self._simple_keyword_fallback(text, is_negative)

    def _simple_keyword_fallback(self, text, is_negative):
        """Fallback keyword extraction if KeyBERT fails"""
        words = re.findall(r'\b\w{4,}\b', text.lower())

        # Enhanced stopwords for Italian
        stopwords = {
            'questo', 'quello', 'molto', 'tutto', 'sempre', 'anche', 'dalla',
            'nella', 'della', 'delle', 'degli', 'sono', 'stata', 'stato',
            'erano', 'aveva', 'avuto', 'fatto', 'dire', 'detto', 'bene',
            'male', 'più', 'meno', 'come', 'quando', 'dove', 'cosa', 'che',
            'nostro', 'vostra', 'essere', 'avere', 'fare', 'andare', 'venire'
        }

        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return keywords[:5] if not is_negative else keywords[:3]

    def analyze_reviews(self, reviews_data):
        """
        Main analysis function - processes all reviews and returns structured insights
        """
        print(f"Analyzing {len(reviews_data)} reviews...")

        results = {
            "summary": {},
            "categories": {},
            "strengths": [],
            "areas_for_improvement": [],
            "detailed_reviews": []
        }

        # Initialize category tracking
        category_sentiments = defaultdict(list)
        category_mentions = defaultdict(int)
        all_positive_keywords = []
        all_negative_keywords = []

        all_sentiments = []

        # Process each review
        for review in reviews_data:
            detailed_review = {
                "title": review.get("titolo", ""),
                "positive_sentiment": None,
                "negative_sentiment": None,
                "categories_mentioned": [],
                "key_issues": [],
                "key_strengths": []
            }

            # Analyze positive content
            pos_content = review.get("contenuto_positivo", "")
            if pos_content:
                pos_analysis = self._analyze_text_sentiment(pos_content)
                detailed_review["positive_sentiment"] = pos_analysis["sentiment"]
                all_sentiments.append(pos_analysis)

                # Extract categories and keywords
                categories = self._categorize_text(pos_content)
                detailed_review["categories_mentioned"].extend(categories)

                keywords = self._extract_keywords(pos_content, is_negative=False)
                detailed_review["key_strengths"] = keywords
                all_positive_keywords.extend(keywords)

                # Track category sentiment
                for cat in categories:
                    category_sentiments[cat].append(pos_analysis)
                    category_mentions[cat] += 1

            # Analyze negative content
            neg_content = review.get("contenuto_negativo")
            if neg_content and neg_content.strip() and neg_content.lower() not in ['null', 'nulla',
                                                                                   'nulla, tutto perfetto!']:
                neg_analysis = self._analyze_text_sentiment(neg_content)
                detailed_review["negative_sentiment"] = neg_analysis["sentiment"]
                all_sentiments.append(neg_analysis)

                keywords = self._extract_keywords(neg_content, is_negative=True)
                detailed_review["key_issues"] = keywords
                all_negative_keywords.extend(keywords)
            else:
                detailed_review["negative_sentiment"] = "None"

            # Remove duplicates from categories
            detailed_review["categories_mentioned"] = list(set(detailed_review["categories_mentioned"]))
            results["detailed_reviews"].append(detailed_review)

        # Calculate summary statistics with original logic
        total_reviews = len(reviews_data)
        positive_reviews = sum(1 for r in results["detailed_reviews"] if r["negative_sentiment"] in ["None", "Neutral"])

        avg_confidence = sum(s["confidence"] for s in all_sentiments) / len(all_sentiments) if all_sentiments else 0

        results["summary"] = {
            "total_reviews": total_reviews,
            "overall_sentiment": "Very Positive" if positive_reviews / total_reviews > 0.8 else "Positive",
            "average_confidence": round(avg_confidence, 3),
            "positive_percentage": round((positive_reviews / total_reviews) * 100, 1)
        }

        # Calculate category insights
        for category, sentiments in category_sentiments.items():
            if sentiments:
                avg_conf = sum(s["confidence"] for s in sentiments) / len(sentiments)
                most_common_sentiment = Counter(s["sentiment"] for s in sentiments).most_common(1)[0][0]

                results["categories"][category] = {
                    "mentions": category_mentions[category],
                    "avg_sentiment": most_common_sentiment,
                    "confidence": round(avg_conf, 3)
                }

        # Generate better insights
        # Top positive aspects (filter out generic words)
        positive_counter = Counter(all_positive_keywords)
        meaningful_strengths = [(k, v) for k, v in positive_counter.most_common(10)
                                if len(k) > 3 and k not in ['tutto', 'molto', 'bene']]

        # Real improvement areas from negative feedback
        improvement_counter = Counter(all_negative_keywords)
        real_issues = [(k, v) for k, v in improvement_counter.most_common(5)
                       if v > 0 and k not in ['meno', 'poco']]

        results["strengths"] = meaningful_strengths[:5]
        results["areas_for_improvement"] = real_issues[:3]

        print("Analysis completed! ✅")
        return results


def main():
    """Demo function to test the analyzer with real hotel reviews"""
    analyzer = BookingReviewAnalyzer()

    # Use the real reviews from constants
    print(f"Analyzing {len(SAMPLE_REVIEWS)} real hotel reviews from Booking.com...")

    # Run analysis
    results = analyzer.analyze_reviews(SAMPLE_REVIEWS)

    # Print comprehensive results to console
    print("\n" + "=" * 60)
    print("🏨 BOOKING REVIEW ANALYSIS RESULTS")
    print("=" * 60)

    # Summary section
    summary = results['summary']
    print(f"\n📊 SUMMARY:")
    print(f"   Total Reviews: {summary['total_reviews']}")
    print(f"   Overall Sentiment: {summary['overall_sentiment']}")
    print(f"   Positive Rate: {summary['positive_percentage']}%")
    print(f"   Average Confidence: {summary['average_confidence']}")

    # Categories section
    print(f"\n🏷️  CATEGORIES ANALYSIS:")
    for category, data in results['categories'].items():
        category_name = category.replace('_', ' ').title()
        print(f"   {category_name}: {data['mentions']} mentions | {data['avg_sentiment']} | Conf: {data['confidence']}")

    # Strengths section
    print(f"\n💪 TOP STRENGTHS:")
    if results['strengths']:
        for i, (strength, count) in enumerate(results['strengths'][:5], 1):
            print(f"   {i}. {strength} (mentioned {count} times)")
    else:
        print("   No specific strengths identified (check keyword extraction)")

    # Areas for improvement
    print(f"\n🎯 AREAS FOR IMPROVEMENT:")
    if results['areas_for_improvement']:
        for i, (issue, count) in enumerate(results['areas_for_improvement'][:3], 1):
            print(f"   {i}. {issue} (mentioned {count} times)")
    else:
        print("   No specific issues identified (check keyword extraction)")

    # Detailed reviews sample
    print(f"\n📝 SAMPLE DETAILED REVIEWS:")
    for i, review in enumerate(results['detailed_reviews'][:3], 1):
        print(f"\n   Review {i}: '{review['title']}'")
        print(f"   Positive: {review['positive_sentiment']}")
        print(f"   Negative: {review['negative_sentiment']}")
        print(f"   Categories: {', '.join(review['categories_mentioned'])}")
        if review['key_strengths']:
            print(f"   Key Strengths: {', '.join(review['key_strengths'][:3])}")
        if review['key_issues']:
            print(f"   Key Issues: {', '.join(review['key_issues'][:3])}")

    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)

    # Return results for potential further processing
    return results


if __name__ == "__main__":
    main()