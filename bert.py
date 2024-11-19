from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch


model = SentenceTransformer('all-MiniLM-L6-v2')  
print("Model Loaded Successfully!")
print("Torch Version:", torch.__version__)

emails = [
    "Subject: Appointment Confirmation - Your appointment is confirmed for tomorrow at 2 PM. Please let us know if you need to reschedule.",
    "Subject: Reminder: Dentist Appointment Tomorrow - This is a reminder about your dentist appointment tomorrow at 10 AM. Please be on time.",
    "Subject: Medical Appointment Rescheduled - We regret to inform you that your medical appointment has been rescheduled to next Wednesday at 3 PM.",
    "Subject: New Appointment Scheduled for Thursday - We have scheduled your next appointment for Thursday at 1 PM. Please mark your calendar.",
    "Subject: Appointment Reminder for Today - Just a reminder about your appointment today at 4 PM. Looking forward to seeing you.",
    "Subject: Medication Refill Reminder - Your prescription for [Medication] is ready for refill. Please visit the pharmacy to pick it up.",
    "Subject: Prescription Ready for Pickup - Your prescribed medication is available for pickup. Kindly come by the pharmacy at your convenience.",
    "Subject: Reminder: Take Your Medications - Just a reminder to take your prescribed medication at 9 AM today. Let us know if you need assistance.",
    "Subject: Refill Your Medication Soon - You’re due for a refill of [Medication]. Please call us to schedule a pickup or have it delivered.",
    "Subject: New Prescription Available - Your new prescription for [Medication] has been approved. You can pick it up at the pharmacy anytime this week.",
    "Subject: You're Invited to Our Annual Gala - Join us for our Annual Gala on December 5th at 7 PM. Please RSVP by November 30th.",
    "Subject: Invitation to Team Building Event - We are organizing a team-building event next Friday at 3 PM. We’d love for you to join us!",
    "Subject: Seminar Invitation: Digital Marketing Trends - You’re invited to a free seminar on digital marketing trends, scheduled for next Monday at 10 AM.",
    "Subject: RSVP: Company Holiday Party - Our company holiday party is coming up on December 20th. Please RSVP by this Friday.",
    "Subject: You're Invited: Charity Fundraiser - We would love for you to join us at the charity fundraiser next Saturday at 6 PM. Your presence will make a difference.",
    "Subject: Flight Booking Confirmation - Your flight to [Destination] has been confirmed for next Monday at 9 AM. Please find your e-ticket attached.",
    "Subject: Hotel Reservation Confirmation - Your hotel booking at [Hotel Name] for your stay in [City] from [Dates] has been confirmed.",
    "Subject: Travel Itinerary for Your Business Trip - Attached is your travel itinerary for the business trip to [City]. Your flight is at 7 AM on Monday.",
    "Subject: Reminder: Airport Shuttle Service - Your airport shuttle service for your flight on [Date] has been arranged. Please be at the hotel lobby by 8 AM.",
    "Subject: Travel Plans for Conference - Your travel and accommodation have been booked for the upcoming conference in [Location] from [Dates].",
    "Subject: Interview Invitation for [Position] - We would like to invite you for an interview for the [Position] role on [Date] at [Time].",
    "Subject: Follow-Up on Your Job Application - Thank you for applying for the [Position]. We’ve reviewed your application and would like to schedule an interview.",
    "Subject: Interview Rescheduled - Due to unforeseen circumstances, we’ve rescheduled your interview for [Position] to [New Date] at [New Time].",
    "Subject: Job Application Status Update - Thank you for your interest in the [Position] role. We’re in the final stages of our hiring process and will update you shortly.",
    "Subject: Job Offer for [Position] - We’re pleased to extend an offer for the [Position] role at our company. Please review the offer details attached."
]



embeddings = model.encode(emails)
embeddings = np.array(embeddings, dtype='float32')
# Initialize FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance for similarity

# Add embeddings to the index
index.add(embeddings)
# Query embedding
query = "will the dentist appointment be tomorrow?"

query_embedding = model.encode([query]).astype('float32')

# Search for top 10 similar emails
distances, indices = index.search(query_embedding, k=5)

# Retrieve the most similar emails
similar_emails = [emails[i] for i in indices[0]]
print(similar_emails)