import type { Metadata } from "next";
import LegalPage from "@/components/marketing/LegalPage";

export const metadata: Metadata = {
  title: "Terms of Service — danta-search",
  description: "The terms that govern your use of danta-search.",
};

export default function TermsPage() {
  return (
    <LegalPage
      label="Legal"
      title="Terms of Service"
      updated="June 2026"
      intro="These Terms of Service govern your access to and use of danta-search. By using the service, you agree to these terms."
      sections={[
        {
          heading: "Use of the service",
          body: [
            "You may use danta-search only in compliance with these terms and all applicable laws.",
            "You are responsible for maintaining the confidentiality of your account credentials and for all activity under your account.",
          ],
        },
        {
          heading: "Your data",
          body: [
            "You retain ownership of the data you upload. You grant us a limited license to process that data solely to provide the service to you.",
            "You are responsible for ensuring you have the right to upload and analyze the data you provide.",
          ],
        },
        {
          heading: "Acceptable use",
          body: [
            "You agree not to misuse the service, including attempting to access other tenants' data, reverse-engineer the platform, or disrupt its operation.",
            "We may suspend or terminate access for violations of these terms.",
          ],
        },
        {
          heading: "Service availability",
          body: [
            "We work to keep the service available and reliable, but it is provided on an \"as is\" and \"as available\" basis without warranties of any kind.",
            "Features may change as the product evolves.",
          ],
        },
        {
          heading: "Limitation of liability",
          body: [
            "To the maximum extent permitted by law, danta-search is not liable for indirect, incidental, or consequential damages arising from your use of the service.",
          ],
        },
        {
          heading: "Changes to these terms",
          body: [
            "We may update these terms from time to time. Continued use of the service after changes take effect constitutes acceptance of the revised terms.",
          ],
        },
      ]}
    />
  );
}
