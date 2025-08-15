import json
import re
from collections import defaultdict, Counter
from transformers import pipeline
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
        print("Model loaded! Ready to analyze guest feedback 🏨")

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

        # Convert to human readable
        sentiment_map = {
            'LABEL_0': 'Very Negative',
            'LABEL_1': 'Negative',
            'LABEL_2': 'Neutral',
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative'
        }

        raw_label = result[0]['label']
        sentiment = sentiment_map.get(raw_label, raw_label)
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

    def _extract_keywords(self, text):
        """Extract relevant keywords from text"""
        if not text:
            return []

        # Simple keyword extraction - could be enhanced with more sophisticated NLP
        words = re.findall(r'\b\w{4,}\b', text.lower())

        # Filter out common words
        stopwords = {'questo', 'quello', 'molto', 'tutto', 'sempre', 'anche', 'dalla', 'dalla', 'nella', 'della'}
        keywords = [word for word in words if word not in stopwords]

        return keywords[:5]  # Return top 5 keywords

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

                keywords = self._extract_keywords(pos_content)
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

                keywords = self._extract_keywords(neg_content)
                detailed_review["key_issues"] = keywords
                all_negative_keywords.extend(keywords)
            else:
                detailed_review["negative_sentiment"] = "None"

            # Remove duplicates from categories
            detailed_review["categories_mentioned"] = list(set(detailed_review["categories_mentioned"]))
            results["detailed_reviews"].append(detailed_review)

        # Calculate summary statistics
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

        # Generate insights
        results["strengths"] = list(Counter(all_positive_keywords).most_common(5))
        results["areas_for_improvement"] = list(Counter(all_negative_keywords).most_common(3))

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
    for i, (strength, count) in enumerate(results['strengths'][:5], 1):
        print(f"   {i}. {strength} (mentioned {count} times)")

    # Areas for improvement
    print(f"\n🎯 AREAS FOR IMPROVEMENT:")
    for i, (issue, count) in enumerate(results['areas_for_improvement'][:3], 1):
        print(f"   {i}. {issue} (mentioned {count} times)")

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