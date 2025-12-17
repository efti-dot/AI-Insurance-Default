from main import process_attachment, query

class InsuranceAssistantService:
    def handle_attachment(self, file):
        return process_attachment(file)

    def handle_query(self, user_input, history):
        return query(user_input, history)