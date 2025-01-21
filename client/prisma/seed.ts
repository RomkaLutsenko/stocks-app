import { PrismaClient } from "@prisma/client";
const prisma = new PrismaClient();
async function main() {
  await prisma.user.create({
    data: {
      id: "0",
      name: "Roman",
      email: "RomanLutsenko@gmail.com",
    },
  });

  await prisma.user.create({
    data: {
      id: "1",
      name: "Dima",
      email: "Dima@gmail.com",
    },
  });
}
main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
