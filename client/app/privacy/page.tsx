import type { Metadata } from "next";
import LegalPage from "@/components/marketing/LegalPage";

export const metadata: Metadata = {
  title: "Privacy Policy — danta-search",
  description: "How danta-search collects, uses, and protects your data.",
};

export default function PrivacyPage() {
  return (
    <LegalPage
      label="Legal"
      title="Privacy Policy"
      updated="June 2026"
      intro="This Privacy Policy explains what information danta-search collects, how we use it, and the choices you have. We design for data isolation and minimal collection by default."
      sections={[
        {
          heading: "Information we collect",
          body: [
            "Account information you provide when signing up, such as your name, work email, and organization.",
            "Data you upload — spreadsheets, documents, and files you choose to analyze with the platform.",
            "Usage and diagnostic data, such as log events, that help us operate and improve the service.",
          ],
        },
        {
          heading: "How we use your information",
          body: [
            "To provide the service: ingesting, indexing, and answering questions over the data you upload.",
            "To maintain security, prevent abuse, and troubleshoot issues.",
            "To communicate with you about your account, updates, and support requests.",
          ],
        },
        {
          heading: "Data isolation and security",
          body: [
            "Each customer's data is isolated by tenant. Storage, search indices, and caches are scoped per organization so one tenant's data cannot reach another's.",
            "Access is governed by role-based access control, and authentication uses industry-standard tokens.",
            "We do not sell your data, and we do not use the contents of your uploaded files to train third-party models.",
          ],
        },
        {
          heading: "Data retention",
          body: [
            "We retain your data for as long as your account is active or as needed to provide the service.",
            "You may request deletion of your data, after which it is removed from active systems within a reasonable period.",
          ],
        },
        {
          heading: "Your rights",
          body: [
            "Depending on your jurisdiction, you may have rights to access, correct, export, or delete your personal data.",
            "To exercise these rights, contact us using the details below.",
          ],
        },
        {
          heading: "Changes to this policy",
          body: [
            "We may update this policy from time to time. Material changes will be communicated through the service or by email.",
          ],
        },
      ]}
    />
  );
}
